import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond_inpaint
from tqdm import tqdm


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


def audio_to_numpy(audio: torch.Tensor) -> np.ndarray:
    # SA3 output is shape (batch, channels, samples). batch_size=1 here, so
    # collapse to (channels, samples) then transpose to (samples, channels)
    # which is what soundfile expects.
    if audio.dim() == 3:
        audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.detach().to(torch.float32).cpu().numpy()
    if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    return np.clip(audio, -1.0, 1.0)


def generate_for_jsonl(jsonl_path: Path, out_root: Path, model, model_config: dict, args):
    """Generate all WAVs for a single JSONL file. Output: <out_root>/<stem>/."""
    out_dir = out_root / jsonl_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = out_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    device = next(model.parameters()).device

    jobs = list(read_jsonl(jsonl_path))
    print(f"Loaded {len(jobs)} jobs from {jsonl_path}")

    for job in tqdm(jobs, desc=f"Generating {jsonl_path.stem}"):
        sample_id = job["id"]
        prompt = job["prompt"]
        duration = float(job.get("duration", args.default_duration))
        seed = int(job.get("seed", 0))
        negative_prompt = job.get("negative_prompt", args.negative_prompt)

        for variant_index in range(args.num_waveforms_per_prompt):
            suffix = f"_v{variant_index:02d}" if args.num_waveforms_per_prompt > 1 else ""
            wav_path = out_dir / f"{sample_id}{suffix}.wav"
            json_path = metadata_dir / f"{sample_id}{suffix}.json"

            if args.skip_existing and wav_path.exists():
                continue

            conditioning = [{"prompt": prompt, "seconds_total": duration}]
            negative_conditioning = (
                [{"prompt": negative_prompt, "seconds_total": duration}]
                if negative_prompt
                else None
            )

            started = time.time()
            output = generate_diffusion_cond_inpaint(
                model,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                sample_size=sample_size,
                seed=seed + variant_index,
                device=device,
                sampler_type=args.sampler,
            )

            audio = audio_to_numpy(output)
            sf.write(wav_path, audio, sample_rate, subtype="PCM_24")

            metadata = {
                "id": sample_id,
                "category": job.get("category"),
                "variant_index": variant_index,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed + variant_index,
                "duration_requested_seconds": duration,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "sampler": args.sampler,
                "model": args.model,
                "sample_rate": sample_rate,
                "output_file": str(wav_path),
                "generation_seconds": round(time.time() - started, 3),
            }
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            del output
            torch.cuda.empty_cache()


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
    parser.add_argument("--num-waveforms-per-prompt", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT,
                        help="Fallback negative prompt if a JSONL row doesn't set one. "
                             "Pass empty string to disable.")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available. Run this on a cloud NVIDIA GPU pod.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, model_config = get_pretrained_model(args.model)
    model = model.to("cuda").to(torch.float16)
    model.eval()
    print(f"sample_rate={model_config['sample_rate']} sample_size={model_config['sample_size']}")

    for jsonl in args.prompts:
        generate_for_jsonl(Path(jsonl), out_root, model, model_config, args)


if __name__ == "__main__":
    main()
