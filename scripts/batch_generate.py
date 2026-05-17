import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from diffusers import StableAudioPipeline
from tqdm import tqdm


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, distorted, noisy, clipped, music loop, drum loop, "
    "hi hats, snare, cymbals, vocals, melody, long ambience, reverb wash"
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


def audio_to_numpy(audio):
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2 and audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
        audio = audio.T

    return np.clip(audio, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True, help="Path to JSONL prompts file")
    parser.add_argument("--out", default="outputs/raw", help="Output directory")
    parser.add_argument("--model", default="stabilityai/stable-audio-open-1.0")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--default-duration", type=float, default=1.5)
    parser.add_argument("--num-waveforms-per-prompt", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available. Run this on a cloud NVIDIA GPU pod.")

    prompts_path = Path(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = out_dir / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    pipe = StableAudioPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")

    sample_rate = pipe.vae.sampling_rate
    jobs = list(read_jsonl(prompts_path))
    print(f"Loaded {len(jobs)} jobs")

    for job in tqdm(jobs, desc="Generating"):
        sample_id = job["id"]
        prompt = job["prompt"]
        duration = float(job.get("duration", args.default_duration))
        seed = int(job.get("seed", 0))

        for variant_index in range(args.num_waveforms_per_prompt):
            suffix = f"_v{variant_index:02d}" if args.num_waveforms_per_prompt > 1 else ""
            wav_path = out_dir / f"{sample_id}{suffix}.wav"
            json_path = metadata_dir / f"{sample_id}{suffix}.json"

            if args.skip_existing and wav_path.exists():
                continue

            generator = torch.Generator("cuda").manual_seed(seed + variant_index)

            started = time.time()
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg_scale,
                audio_end_in_s=duration,
                num_waveforms_per_prompt=1,
                generator=generator,
            )

            audio = audio_to_numpy(result.audios[0])
            sf.write(wav_path, audio, sample_rate, subtype="PCM_24")

            metadata = {
                "id": sample_id,
                "variant_index": variant_index,
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "seed": seed + variant_index,
                "duration_requested_seconds": duration,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "model": args.model,
                "sample_rate": sample_rate,
                "output_file": str(wav_path),
                "generation_seconds": round(time.time() - started, 3),
            }
            json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            del result
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
