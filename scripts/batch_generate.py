"""Batched Stable Audio 3 generation for the drum + pitched pipelines.

Loads the model ONCE and generates every (prompt × variant) across all the
given JSONL files. Generation is batched (default 16, push to 32-64 on an
80GB GPU) — one `generate_diffusion_cond_inpaint` call per batch instead of
one per sample, which is the big throughput win on a large GPU.

Key behaviours:
  - Per-row variant counts: each JSONL row may carry a "variants" field
    (written by list_to_jsonl_pitched.py from the category config); rows
    without it fall back to --num-waveforms-per-prompt. This lets hard/weak
    categories oversample more without a global flag.
  - Content-addressed --skip-existing: units whose output WAV already exists
    are dropped before batching (resumable runs).
  - Duration bucketing: batches are homogeneous in seconds_total (and, when
    --init-audio-anchor is on, in target pitch — the anchor tone broadcasts
    across a batch, so an anchored batch must share one pitch).
  - GPU pre-gate: NaN/Inf/silent/clipped outputs are dropped before they hit
    disk, so the (unbatched, expensive) gate stage never sees obvious failures.
  - init_audio pitch anchoring (EXPERIMENT, default off): seeds generation with
    a synthesized harmonic tone at the row's target pitch to bias SA3 toward
    the right pitch. A/B it on a slice before trusting it (see the v3 plan §1d);
    it can collapse timbral variety if init_noise_level is too low.

NOTE: batch_size is part of a run's identity — a not-yet-generated variant's
noise comes from a batched RNG draw, so changing --batch-size changes the audio
of new variants. Already-written WAVs are never regenerated (skip-existing), so
resumed runs are stable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# torch + stable_audio_tools are imported lazily inside generate_all()/main() so
# this module imports on a CPU box without CUDA wheels (lets us unit-test the
# pure-Python work-list / pre-gate logic locally).


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "vocals, melody, long ambience, reverb wash"
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def item_to_numpy(item) -> np.ndarray:
    """One batch item [channels, samples] -> [samples, channels] float32, clipped.
    `item` is a torch.Tensor; we call methods on it (no torch import needed)."""
    a = item.detach().float().cpu().numpy()
    if a.ndim == 2 and a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
        a = a.T
    return np.clip(a, -1.0, 1.0)


def pregate_reason(a: np.ndarray) -> str | None:
    """Cheap reject of obviously-bad output before it hits disk / the gate.
    Returns a reason string, or None if the audio looks usable."""
    if not np.all(np.isfinite(a)):
        return "nan_or_inf"
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    if peak < 1e-4:  # < ~-80 dBFS — effectively silent
        return "silent"
    if float(np.mean(np.abs(a) >= 0.999)) > 0.01:  # >1% of samples pinned to the rails
        return "clipped"
    return None


def synth_anchor_tone(target_midi: int, sr: int, n_samples: int, channels: int):
    """A harmonic tone at `target_midi` to use as init_audio (pitch anchor).
    Returns a torch.Tensor [channels, n_samples], peak ~0.5. Decaying partials
    so it reads as a pitched note, not a pure sine."""
    import torch
    f0 = 440.0 * (2.0 ** ((target_midi - 69) / 12.0))
    t = np.arange(n_samples, dtype=np.float32) / sr
    y = np.zeros(n_samples, dtype=np.float32)
    for h, amp in ((1, 1.0), (2, 0.5), (3, 0.33), (4, 0.25)):
        y += (amp * np.sin(2.0 * np.pi * f0 * h * t)).astype(np.float32)
    m = float(np.max(np.abs(y))) or 1.0
    y = (0.5 * y / m).astype(np.float32)
    arr = np.tile(y, (max(1, channels), 1))
    return torch.from_numpy(arr)


def build_units(jsonl_path: Path, out_root: Path, args) -> tuple[Path, Path, list[dict]]:
    """Expand a JSONL into a flat work-list of generation units (one per
    prompt×variant), applying --skip-existing up front."""
    out_dir = out_root / jsonl_path.stem
    metadata_dir = out_dir / "_metadata"
    units: list[dict] = []
    for job in read_jsonl(jsonl_path):
        sid = job["id"]
        prompt = job["prompt"]
        duration = float(job.get("duration", args.default_duration))
        seed = int(job.get("seed", 0))
        negative_prompt = job.get("negative_prompt", args.negative_prompt)
        target_pitch = job.get("target_pitch_midi")
        nvar = int(job.get("variants", args.num_waveforms_per_prompt))
        for variant_index in range(max(1, nvar)):
            suffix = f"_v{variant_index:02d}" if nvar > 1 else ""
            wav_path = out_dir / f"{sid}{suffix}.wav"
            if args.skip_existing and wav_path.exists():
                continue
            units.append({
                "id": sid,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "duration": duration,
                "nominal_seed": seed + variant_index,
                "target_pitch": target_pitch,
                "variant_index": variant_index,
                "category": job.get("category"),
                "wav_path": wav_path,
                "json_path": metadata_dir / f"{sid}{suffix}.json",
            })
    return out_dir, metadata_dir, units


def generate_all(jsonls, out_root: Path, model, model_config: dict, args) -> None:
    import torch
    from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    channels = int(model_config.get("audio_channels", 2))
    device = next(model.parameters()).device

    # Collect units across ALL categories so a batch can fill from same-duration
    # rows of different categories (better packing).
    all_units: list[dict] = []
    for jp in jsonls:
        out_dir, metadata_dir, units = build_units(Path(jp), out_root, args)
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        print(f"[batch_generate] {Path(jp).stem}: {len(units)} units to generate")
        all_units.extend(units)

    if not all_units:
        print("[batch_generate] nothing to generate (all skipped / empty)")
        return

    # Bucket by (duration, anchor-pitch). Anchor pitch only matters when the
    # init_audio anchor is on (one anchor tone is shared across the batch).
    def bucket_key(u: dict):
        return (round(u["duration"], 4), u["target_pitch"] if args.init_audio_anchor else None)

    buckets: dict = defaultdict(list)
    for u in all_units:
        buckets[bucket_key(u)].append(u)

    batches: list[tuple] = []
    for key, ulist in buckets.items():
        for i in range(0, len(ulist), args.batch_size):
            batches.append((key, ulist[i:i + args.batch_size]))

    print(f"[batch_generate] {len(all_units)} units -> {len(batches)} batches "
          f"(batch_size={args.batch_size}, anchor={'on' if args.init_audio_anchor else 'off'})")

    written = 0
    dropped = 0
    for bidx, (key, batch) in enumerate(tqdm(batches, desc="Generating")):
        conditioning = [{"prompt": u["prompt"], "seconds_total": u["duration"]} for u in batch]
        negative_conditioning = [
            {"prompt": (u["negative_prompt"] or ""), "seconds_total": u["duration"]} for u in batch
        ]
        batch_seed = 2001 + bidx

        gen_kwargs = dict(
            model=model,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            sample_size=sample_size,
            seed=batch_seed,
            device=device,
            sampler_type=args.sampler,
            batch_size=len(batch),
        )

        anchor_pitch = key[1]
        used_anchor = False
        if args.init_audio_anchor and anchor_pitch is not None:
            # EXPERIMENT: one anchor tone broadcasts across the (single-pitch) batch.
            try:
                anchor = synth_anchor_tone(int(anchor_pitch), sample_rate, sample_size, channels)
                gen_kwargs["init_audio"] = (sample_rate, anchor)
                gen_kwargs["init_noise_level"] = args.init_noise_level
                used_anchor = True
            except Exception as e:  # noqa: BLE001
                print(f"[batch_generate] anchor synth failed (pitch {anchor_pitch}): {e}", file=sys.stderr)

        started = time.time()
        output = generate_diffusion_cond_inpaint(**gen_kwargs)
        elapsed = time.time() - started
        if output.dim() == 2:  # defensive: single-item shape [C, N]
            output = output.unsqueeze(0)

        for i, u in enumerate(batch):
            audio = item_to_numpy(output[i])
            bad = pregate_reason(audio)
            if bad is not None:
                dropped += 1
                continue
            sf.write(str(u["wav_path"]), audio, sample_rate, subtype="PCM_24")
            metadata = {
                "id": u["id"],
                "category": u["category"],
                "variant_index": u["variant_index"],
                "prompt": u["prompt"],
                "negative_prompt": u["negative_prompt"],
                "seed": batch_seed,
                "batch_index": bidx,
                "batch_position": i,
                "batch_size": len(batch),
                "duration_requested_seconds": u["duration"],
                "target_pitch_midi": u["target_pitch"],
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "sampler": args.sampler,
                "model": args.model,
                "sample_rate": sample_rate,
                "init_audio_anchor": used_anchor,
                "init_noise_level": args.init_noise_level if used_anchor else None,
                "output_file": str(u["wav_path"]),
                "generation_seconds_batch": round(elapsed, 3),
            }
            u["json_path"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            written += 1

        del output
        torch.cuda.empty_cache()

    print(f"[batch_generate] done. written={written} dropped_pregate={dropped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, nargs="+",
                        help="One or more JSONL prompts files. Output for each goes "
                             "to <out-root>/<jsonl-stem>/")
    parser.add_argument("--out-root", default="outputs/raw",
                        help="Per-category output root. Default: outputs/raw")
    parser.add_argument("--model", default="stabilityai/stable-audio-3-medium",
                        help="HF repo id. Alternatives: "
                             "stabilityai/stable-audio-3-small-sfx (0.6B, SFX-tuned), "
                             "stabilityai/stable-audio-3-small-music (0.6B, music).")
    parser.add_argument("--steps", type=int, default=8,
                        help="Diffusion steps. SA3 converges in ~8 (vs 120 for SAO 1.0).")
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Classifier-free guidance. SA3 defaults to 1.0 (vs 7.0 for SAO 1.0).")
    parser.add_argument("--sampler", default="pingpong",
                        help="Sampler type. SA3 model card uses 'pingpong'.")
    parser.add_argument("--default-duration", type=float, default=1.5)
    parser.add_argument("--num-waveforms-per-prompt", type=int, default=1,
                        help="Fallback variant count for rows without a 'variants' field "
                             "(pitched JSONLs set per-row 'variants' from the category config).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Generations per model call. 16 default; 32-64 on an 80GB GPU.")
    parser.add_argument("--init-audio-anchor", action="store_true",
                        help="EXPERIMENT: seed generation with a synth tone at the row's "
                             "target pitch to bias SA3 toward correct pitch. A/B before trusting.")
    parser.add_argument("--init-noise-level", type=float, default=0.7,
                        help="init_audio noise level (1.0=ignore anchor, lower=more anchor). "
                             "Only used with --init-audio-anchor.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT,
                        help="Fallback negative prompt if a JSONL row doesn't set one. "
                             "Pass empty string to disable.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    import torch
    from stable_audio_tools import get_pretrained_model

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available. Run this on a cloud NVIDIA GPU pod.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, model_config = get_pretrained_model(args.model)
    model = model.to("cuda").to(torch.float16)
    model.eval()
    print(f"sample_rate={model_config['sample_rate']} sample_size={model_config['sample_size']}")

    generate_all(args.prompts, out_root, model, model_config, args)


if __name__ == "__main__":
    main()
