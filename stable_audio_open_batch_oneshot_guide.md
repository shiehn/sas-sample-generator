# Stable Audio Open Batch One-Shot Generator Guide

**Goal:** generate large batches of one-shot audio samples, for example: 1,000 kick drum prompts in → 1,000 generated kick drum WAVs out.

**Target user setup:** Apple Silicon MacBook Air for control, editing, curation, and downloads; temporary cloud GPU for generation; no physical GPU purchase; no permanently reserved AWS GPU.

---

## 0. Recommended architecture

```text
Your M4 MacBook Air
  ├─ write/edit prompts.jsonl
  ├─ SSH/Jupyter into temporary GPU machine
  ├─ download final ZIP of WAVs + metadata
  └─ audition/curate/import samples into Signals & Sorcery

Temporary cloud GPU machine
  ├─ loads Stable Audio Open
  ├─ runs batch_generate.py
  ├─ runs postprocess_oneshots.py
  ├─ writes WAV files and metadata
  └─ shuts down when finished
```

**Core recommendation:** use a temporary RunPod GPU pod with an **L40S 48GB** GPU for the first serious run. Use Colab only for a quick smoke test or if you want the easiest notebook experience.

---

## 1. Why Stable Audio Open for this job

Stable Audio Open 1.0 is a text-to-audio model intended for generation of music/audio from prompts. It can generate variable-length stereo audio at 44.1kHz, up to 47 seconds. For one-shots, you will deliberately request short durations such as 1.0–2.5 seconds.

This is a better fit than a general audio conversation model because your target workflow is:

```text
prompt → generated audio sample
```

not:

```text
audio conversation / speech understanding / transcription
```

---

## 2. Account checklist

Create or confirm these accounts:

1. **Hugging Face account**
   - Required to download `stabilityai/stable-audio-open-1.0`.
   - You must accept the model license/terms on the Hugging Face model page before the download works.

2. **RunPod account**
   - Recommended for actual batch runs.
   - Use on-demand pods; do not reserve a GPU long-term.

3. **Optional: Google Colab**
   - Useful for a first notebook test.
   - Not recommended as the main automation/batch pipeline because runtimes and GPU availability are not guaranteed.

---

## 3. Cloud GPU choice

### Recommended first choice

Use:

```text
RunPod
GPU: L40S 48GB
Disk/volume: 100–150GB
Mode: on-demand pod
```

Why:

- Enough VRAM headroom.
- You can stop it when finished.
- You only pay while the pod is running.
- Stable Audio generation is GPU-heavy and CUDA-oriented.

### Cheaper test option

After you prove the pipeline on L40S, test:

```text
RunPod
GPU: RTX 4090 24GB
```

It may be cheaper, but the lower VRAM gives you less room for long duration, multiple waveforms per prompt, or heavier settings.

### Colab position

Use Colab for:

- basic smoke test
- notebook experimentation
- checking that prompts sound reasonable

Do not rely on Colab for:

- repeatable 1,000-sample jobs
- unattended batch service
- production-like workflow

---

## 4. Suggested repo layout

Create this project locally on your Mac:

```text
stable-audio-oneshots/
  README.md
  requirements.txt
  prompts/
    kicks_1000.jsonl
  scripts/
    batch_generate.py
    postprocess_oneshots.py
    benchmark.py
  outputs/
    raw/
    processed/
    rejected/
    manifests/
```

You can then copy this folder to the cloud GPU machine.

---

## 5. Prompt file format

Use JSONL: one JSON object per line.

Example: `prompts/kicks_1000.jsonl`

```jsonl
{"id":"kick_0001","prompt":"short punchy analog kick drum one shot, dry, deep sub tail, clean transient, no hi hats, no snare, no cymbals, studio quality","seed":1001,"duration":1.5}
{"id":"kick_0002","prompt":"tight 909-style kick drum one shot, hard transient, short decay, clean low end, no melody, no loop","seed":1002,"duration":1.25}
{"id":"kick_0003","prompt":"deep techno kick drum one shot, warm saturated low end, short click transient, mono-compatible, dry, no percussion loop","seed":1003,"duration":1.75}
```

Recommended prompt rules:

- Say **one shot** explicitly.
- Say **no loop** explicitly.
- Exclude unwanted instruments: hi-hats, snare, cymbals, vocals, melody.
- Keep duration short.
- Use seed values so runs are reproducible.
- Generate more than you need. For example, generate 1,500 and keep the best 1,000.

---

## 6. `requirements.txt`

Create:

```txt
torch
torchaudio
diffusers
transformers
accelerate
safetensors
soundfile
numpy
pandas
tqdm
huggingface_hub
```

On CUDA machines, you may want to install PyTorch from the CUDA wheel index instead of relying on the default PyPI resolution. See the setup commands below.

---

## 7. RunPod setup

### 7.1 Create the pod

In RunPod:

1. Create a new GPU pod.
2. Choose **L40S 48GB**.
3. Use a PyTorch CUDA template if available.
4. Attach a persistent volume, around **100–150GB**.
5. Enable SSH or use the web terminal.
6. Start the pod.

### 7.2 Connect from your Mac

From your Mac terminal:

```bash
ssh root@YOUR_RUNPOD_HOST -p YOUR_RUNPOD_PORT
```

Or use the RunPod web terminal.

### 7.3 Create project directory

On the pod:

```bash
mkdir -p /workspace/stable-audio-oneshots
cd /workspace/stable-audio-oneshots
```

Copy your local project to the pod. From your Mac:

```bash
scp -P YOUR_RUNPOD_PORT -r ./stable-audio-oneshots root@YOUR_RUNPOD_HOST:/workspace/
```

---

## 8. Python environment setup on the pod

From the pod:

```bash
cd /workspace/stable-audio-oneshots

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
```

Install PyTorch with CUDA. The exact CUDA wheel may depend on the RunPod template. A common option is:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install the rest:

```bash
pip install -r requirements.txt
```

Check CUDA:

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected result:

```text
CUDA available: True
Device: NVIDIA L40S
```

---

## 9. Hugging Face login

On your Mac or in a browser:

1. Go to Hugging Face.
2. Create an access token.
3. Accept the Stable Audio Open model terms on the model page.

On the pod:

```bash
huggingface-cli login
```

Paste your token.

---

## 10. Batch generation script

Create `scripts/batch_generate.py`.

```python
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
    """Convert pipeline audio output to shape: samples x channels."""
    if torch.is_tensor(audio):
        audio = audio.detach().float().cpu().numpy()

    audio = np.asarray(audio, dtype=np.float32)

    # Common model output: channels x samples. soundfile wants samples x channels.
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

            # Keep VRAM tidy across long runs.
            del result
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
```

---

## 11. Post-processing script

This trims silence, normalizes, optionally converts to mono, rejects extremely quiet files, and writes a manifest.

Create `scripts/postprocess_oneshots.py`.

```python
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def db_to_amp(db):
    return 10 ** (db / 20)


def ensure_2d(y):
    if y.ndim == 1:
        return y[:, None]
    return y


def apply_fade(y, sr, fade_ms=5):
    fade_len = int(sr * fade_ms / 1000)
    if fade_len <= 1 or len(y) < fade_len * 2:
        return y

    fade_in = np.linspace(0, 1, fade_len)[:, None]
    fade_out = np.linspace(1, 0, fade_len)[:, None]

    y[:fade_len] *= fade_in
    y[-fade_len:] *= fade_out
    return y


def trim_silence(y, sr, threshold_db=-45, pad_ms=15):
    y = ensure_2d(y)
    mono = np.mean(y, axis=1)

    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak < 1e-5:
        return None, "too_quiet"

    threshold = peak * db_to_amp(threshold_db)
    active = np.where(np.abs(mono) >= threshold)[0]

    if len(active) == 0:
        return None, "silence"

    pad = int(sr * pad_ms / 1000)
    start = max(0, int(active[0]) - pad)
    end = min(len(y), int(active[-1]) + pad)

    return y[start:end], None


def normalize_peak(y, target_db=-1.0):
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak < 1e-5:
        return None
    target = db_to_amp(target_db)
    return np.clip(y / peak * target, -1.0, 1.0)


def process_file(path, out_dir, rejected_dir, args):
    y, sr = sf.read(path, always_2d=True)

    trimmed, reject_reason = trim_silence(
        y,
        sr,
        threshold_db=args.trim_threshold_db,
        pad_ms=args.pad_ms,
    )

    if reject_reason:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        return {
            "file": path.name,
            "status": "rejected",
            "reason": reject_reason,
            "duration_seconds": 0,
            "output": str(rejected_path),
        }

    max_samples = int(args.max_seconds * sr)
    if len(trimmed) > max_samples:
        # For one-shots, keep the attack and early body.
        trimmed = trimmed[:max_samples]

    if args.mono:
        trimmed = np.mean(trimmed, axis=1, keepdims=True)

    processed = normalize_peak(trimmed, target_db=args.target_peak_db)
    if processed is None:
        rejected_path = rejected_dir / path.name
        shutil.copy2(path, rejected_path)
        return {
            "file": path.name,
            "status": "rejected",
            "reason": "normalization_failed",
            "duration_seconds": 0,
            "output": str(rejected_path),
        }

    processed = apply_fade(processed, sr, fade_ms=args.fade_ms)

    out_path = out_dir / path.name
    sf.write(out_path, processed, sr, subtype="PCM_24")

    return {
        "file": path.name,
        "status": "ok",
        "reason": "",
        "duration_seconds": round(len(processed) / sr, 4),
        "sample_rate": sr,
        "channels": processed.shape[1],
        "output": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="outputs/raw")
    parser.add_argument("--out-dir", default="outputs/processed")
    parser.add_argument("--rejected-dir", default="outputs/rejected")
    parser.add_argument("--manifest", default="outputs/manifests/processed_manifest.csv")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=2.5)
    parser.add_argument("--trim-threshold-db", type=float, default=-45)
    parser.add_argument("--pad-ms", type=float, default=15)
    parser.add_argument("--fade-ms", type=float, default=5)
    parser.add_argument("--target-peak-db", type=float, default=-1.0)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    rejected_dir = Path(args.rejected_dir)
    manifest_path = Path(args.manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    wavs = sorted(in_dir.glob("*.wav"))
    rows = []

    for wav in tqdm(wavs, desc="Post-processing"):
        rows.append(process_file(wav, out_dir, rejected_dir, args))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row["status"] == "ok")
    rejected = len(rows) - ok
    print(f"Processed: {ok}")
    print(f"Rejected: {rejected}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
```

---

## 12. Smoke test

Before generating 1,000 samples, create `prompts/kicks_smoke_test.jsonl`:

```jsonl
{"id":"kick_test_0001","prompt":"short punchy analog kick drum one shot, dry, deep sub tail, clean transient, no hi hats, no snare, no cymbals, studio quality","seed":1,"duration":1.5}
{"id":"kick_test_0002","prompt":"tight 909-style kick drum one shot, hard transient, short decay, clean low end, no melody, no loop","seed":2,"duration":1.25}
{"id":"kick_test_0003","prompt":"deep techno kick drum one shot, warm saturated low end, short click transient, mono-compatible, dry, no percussion loop","seed":3,"duration":1.75}
```

Run:

```bash
source .venv/bin/activate

python scripts/batch_generate.py \
  --prompts prompts/kicks_smoke_test.jsonl \
  --out outputs/raw \
  --steps 80 \
  --cfg-scale 7 \
  --skip-existing
```

Post-process:

```bash
python scripts/postprocess_oneshots.py \
  --in-dir outputs/raw \
  --out-dir outputs/processed \
  --rejected-dir outputs/rejected \
  --manifest outputs/manifests/smoke_test_manifest.csv \
  --mono \
  --max-seconds 2.5
```

Zip and download:

```bash
zip -r smoke_test_outputs.zip outputs/processed outputs/manifests
```

From your Mac:

```bash
scp -P YOUR_RUNPOD_PORT root@YOUR_RUNPOD_HOST:/workspace/stable-audio-oneshots/smoke_test_outputs.zip .
```

Listen to the files locally before doing a big run.

---

## 13. Benchmark cost before the full run

Create `scripts/benchmark.py`.

```python
import csv
import statistics
from pathlib import Path


def main():
    metadata_dir = Path("outputs/raw/_metadata")
    times = []

    for path in metadata_dir.glob("*.json"):
        import json
        data = json.loads(path.read_text())
        if "generation_seconds" in data:
            times.append(float(data["generation_seconds"]))

    if not times:
        print("No generation metadata found.")
        return

    avg = statistics.mean(times)
    print(f"Samples measured: {len(times)}")
    print(f"Average seconds/sample: {avg:.2f}")

    for hourly_price in [0.86, 0.69, 1.80]:
        estimated_hours = avg * 1000 / 3600
        estimated_cost = estimated_hours * hourly_price
        print(f"At ${hourly_price}/hr: ~{estimated_hours:.2f} hours, ~${estimated_cost:.2f}")


if __name__ == "__main__":
    main()
```

Run:

```bash
python scripts/benchmark.py
```

This gives you a realistic estimate based on your actual pod, settings, prompt durations, and inference steps.

---

## 14. Full batch run

Once the smoke test sounds promising:

```bash
source .venv/bin/activate

python scripts/batch_generate.py \
  --prompts prompts/kicks_1000.jsonl \
  --out outputs/raw \
  --steps 120 \
  --cfg-scale 7 \
  --default-duration 1.5 \
  --skip-existing
```

Post-process:

```bash
python scripts/postprocess_oneshots.py \
  --in-dir outputs/raw \
  --out-dir outputs/processed \
  --rejected-dir outputs/rejected \
  --manifest outputs/manifests/kicks_1000_manifest.csv \
  --mono \
  --max-seconds 2.5
```

Create final ZIP:

```bash
zip -r stable_audio_kicks_1000.zip outputs/processed outputs/manifests outputs/raw/_metadata
```

Download to your Mac:

```bash
scp -P YOUR_RUNPOD_PORT root@YOUR_RUNPOD_HOST:/workspace/stable-audio-oneshots/stable_audio_kicks_1000.zip .
```

Then stop the pod.

---

## 15. Better workflow: generate extra and curate

For usable sample libraries, do not expect all 1,000 generations to be keepers.

Recommended approach:

```text
Generate 1,500 raw kicks
Post-process automatically
Reject silence/bad files automatically
Audition and manually curate
Keep the best 1,000
```

You can also generate categories:

```text
250 deep techno kicks
250 808 kicks
250 909-style kicks
250 acoustic-style electronic kicks
250 distorted industrial kicks
250 short clicky kicks
```

This will give you a more useful sample library than 1,000 near-identical prompts.

---

## 16. Prompt template ideas

### Dry electronic kick

```text
short dry electronic kick drum one shot, punchy transient, controlled low end, clean studio sample, no hi hats, no snare, no cymbals, no melody, no loop
```

### 808-style kick

```text
deep 808-style kick drum one shot, long sub bass decay, smooth sine low end, clean transient, dry, no percussion loop, no vocals, no melody
```

### 909-style kick

```text
tight 909-style kick drum one shot, hard click transient, short punchy body, techno production sample, dry, no hi hats, no snare, no loop
```

### Distorted techno kick

```text
distorted industrial techno kick drum one shot, saturated low mid body, aggressive transient, mono-compatible, dry, no cymbals, no snare, no loop
```

### Soft lo-fi kick

```text
warm lo-fi kick drum one shot, soft transient, dusty sampler texture, short decay, dry, no hi hats, no snare, no melody
```

---

## 17. Practical settings to try

Start conservative:

```text
duration: 1.25–2.0 seconds
steps: 80 for smoke tests
steps: 120 for batch tests
cfg-scale: 6–8
num-waveforms-per-prompt: 1
```

Then experiment:

```text
steps: 150–200 if quality improves enough to justify cost
duration: 2.5 seconds for long 808 tails
num-waveforms-per-prompt: 2–3 for curation runs
```

Avoid starting with maximum settings. First find the cheapest settings that produce usable samples.

---

## 18. When to use RunPod vs Colab

### Use RunPod when

- You want repeatable batch runs.
- You want SSH and a normal project folder.
- You want to stop/start without losing project state if using a persistent volume.
- You want control over GPU type.

### Use Colab when

- You want a fast notebook experiment.
- You are okay with runtime interruption.
- You are not running an unattended 1,000-sample job.

---

## 19. Cost-control rules

1. **Never leave the pod running overnight by accident.**
2. **Benchmark with 10–25 samples first.**
3. **Use `--skip-existing` so resumed jobs do not regenerate paid work.**
4. **Keep a persistent volume for model cache, but delete old raw outputs if storage grows.**
5. **Stop the pod immediately after downloading results.**
6. **Generate short one-shots, not 30–47 second clips.**
7. **Use 80–120 steps first, then only increase if quality requires it.**

Cost formula:

```text
estimated_cost =
  average_generation_seconds_per_sample
  × number_of_samples
  ÷ 3600
  × hourly_gpu_price
```

---

## 20. Common failures

### `CUDA available: False`

You are not on a GPU runtime, or the PyTorch install does not match CUDA.

Fix:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If `nvidia-smi` fails, the machine does not expose an NVIDIA GPU.

### Hugging Face download denied

You probably did not accept the model license terms or your token is not logged in.

Fix:

```bash
huggingface-cli login
```

Then visit the model page in a browser and accept the terms.

### Out of memory

Try:

```text
duration: lower it
steps: lower it
num-waveforms-per-prompt: 1
GPU: use L40S 48GB instead of 24GB GPU
```

### Samples sound like loops

Tighten prompts:

```text
one shot, isolated drum hit, no rhythm, no loop, no hi hats, no snare, no cymbals, no melody
```

### Samples have too much reverb

Add to prompt:

```text
dry, close-mic, no reverb, no ambience
```

Add to negative prompt:

```text
reverb wash, long ambience, room sound
```

---

## 21. Optional next step: small local control CLI

Once this works, wrap the remote generation in a tiny CLI on your Mac:

```bash
sas-samplegen upload prompts/kicks_1000.jsonl
sas-samplegen run --gpu l40s
sas-samplegen download stable_audio_kicks_1000.zip
sas-samplegen stop
```

Internally, that CLI can just SSH/SCP into RunPod at first. Later, you can turn it into a real service if you need.

For your current budget and use case, **do not build a permanently hosted API yet**. Build a reliable batch job first.

---

## 22. Final recommended setup

Use this as your default workflow:

```text
1. Write JSONL prompts on M4 MacBook Air.
2. Start RunPod L40S pod.
3. SSH into pod.
4. Install environment or reuse persistent volume.
5. Run 10–25 sample smoke test.
6. Download and audition.
7. Adjust prompt templates.
8. Run 1,000–1,500 sample batch.
9. Post-process and write manifest.
10. ZIP outputs.
11. Download to Mac.
12. Stop pod.
13. Curate final sample library locally.
```

This gives you cloud GPU power only when you need it, without buying hardware or paying for a permanently reserved instance.

---

## References

- Stable Audio Open 1.0 model card: https://huggingface.co/stabilityai/stable-audio-open-1.0
- Stable Audio Open paper: https://arxiv.org/abs/2407.14358
- Stable Audio Tools: https://github.com/Stability-AI/stable-audio-tools
- RunPod GPU pricing: https://www.runpod.io/pricing
- RunPod L40S page: https://www.runpod.io/gpu-models/l40s
- RunPod RTX 4090 page: https://www.runpod.io/gpu-models/rtx-4090
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
