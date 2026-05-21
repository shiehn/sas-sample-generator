# Stable Audio 3 Batch One-Shot Generator Guide

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
  ├─ loads Stable Audio 3
  ├─ runs batch_generate.py
  ├─ runs postprocess_oneshots.py
  ├─ writes WAV files and metadata
  └─ shuts down when finished
```

**Core recommendation:** use a temporary RunPod GPU pod with an **L40S 48GB** GPU for the first serious run. Use Colab only for a quick smoke test or if you want the easiest notebook experience.

---

## 1. Why Stable Audio 3 for this job

Stable Audio 3 (released May 2026) is a family of text-to-audio latent
diffusion transformers. It supersedes Stable Audio Open 1.0 — the headline
practical difference for this repo is that SA3 converges in ~8 diffusion
steps (vs 120 for SAO 1.0) and so generates one-shots ~15× faster on the
same hardware. The relevant open-weights variants are:

| Repo | Params | Best for |
|---|---|---|
| `stabilityai/stable-audio-3-medium` | 2B | Full musical / textural content (this repo's default). |
| `stabilityai/stable-audio-3-small-sfx` | 0.6B | Pure sound-effect / drum / percussion one-shots. Fastest. |
| `stabilityai/stable-audio-3-small-music` | 0.6B | Short music — not relevant here. |

Switch models via `--model` to `batch_generate.py`. The `small-sfx` variant
is the closest match to the drum / percussion focus of this repo and will
run faster on cheaper GPUs; the default `medium` produces broader, more
"musical" textures. Both generate variable-length stereo audio at the
sample rate reported by `model_config["sample_rate"]` (44.1 kHz).

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
   - Required to download `stabilityai/stable-audio-3-medium` (or whichever
     SA3 variant you point `--model` at).
   - You must accept the SA3 community license and the Gemma terms it
     inherits on the Hugging Face model page before the download works.

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

> **Operational note:** the [`README.md`](README.md) is the source of truth
> for the step-by-step run procedure. This guide focuses on rationale and
> design. If anything in the sections below conflicts with the README,
> follow the README.

The actual layout in this repo (multi-category):

```text
sas-sample-generator/
  README.md
  stable_audio_open_batch_oneshot_guide.md   (this file)
  requirements.txt
  prompts/
    kick.txt           kick.jsonl
    snare-standard.txt snare-standard.jsonl
    snare-rim.txt      snare-rim.jsonl
    hat-closed.txt     hat-closed.jsonl
    hat-open.txt       hat-open.jsonl
    cymbal-ride.txt    cymbal-ride.jsonl
    cymbal-crash.txt   cymbal-crash.jsonl
    cymbal-splash.txt  cymbal-splash.jsonl
    tamborine.txt      tamborine.jsonl
    shaker.txt         shaker.jsonl
    tom-hi.txt         tom-hi.jsonl
    tom-mid.txt        tom-mid.jsonl
    tom-low.txt        tom-low.jsonl
    hit.txt            hit.jsonl
  scripts/
    setup.sh             (pod bootstrap)
    run_all.sh           (full pipeline wrapper)
    categories.txt       (which categories are enabled)
    category_config.py   (per-category negative prompts + durations)
    list_to_jsonl.py     (.txt -> .jsonl converter)
    batch_generate.py
    postprocess_oneshots.py
    benchmark.py
    sync.sh              (optional rclone push)
  outputs/                                       (gitignored)
    raw/<category>/<id>.wav
    raw/<category>/_metadata/<id>.json
    processed/<category>/<id>.wav
    rejected/<category>/<id>.wav
    manifests/<category>.csv
```

Two key changes vs. earlier versions of this doc:

1. **Per-category subdirectories** under `outputs/` — generated WAVs are
   organised by category rather than dumped flat. The wrapper script
   `run_all.sh` orchestrates the whole flow.
2. **Content-addressed IDs** — filenames are `{category}-{hash8}.wav`
   instead of sequential `kick_0001.wav`. The hash is reproducible
   (`sha1(category:prompt:seed)`), which means re-runs are idempotent: a
   prompt that already produced a WAV is skipped by `--skip-existing`.

---

## 5. Prompt file format

**Two layers:**

- **Editable input** (`prompts/<category>.txt`): one description per line,
  blank lines and `#` comments ignored. This is what you write.
- **Generated** (`prompts/<category>.jsonl`): one JSON row per prompt with
  resolved `id`, `category`, `prompt`, `negative_prompt`, `seed`, and
  `duration`. Built automatically by `scripts/list_to_jsonl.py` (which
  `run_all.sh` calls for each category).

Example JSONL row:

```jsonl
{"id":"kick-c1da23da","category":"kick","prompt":"tight 909-style kick drum one shot, hard click transient, short punchy body, dry, no hi hats, no snare, no cymbals, no loop","negative_prompt":"low quality, distorted, ...","seed":1001,"duration":1.5}
```

You normally won't read or write JSONL by hand. Edit the `.txt`.

### Prompt rules (target → positive prompt)

- Say **one shot** explicitly.
- Say **no loop** explicitly.
- Exclude unwanted instruments in the positive prompt where genre-specific:
  `no hi hats, no cymbals, no melody`.
- Keep descriptions ~10–15 words.
- Generate more than you need; curate the keepers locally.

### Per-category negative prompts

The negative prompt lives in `scripts/category_config.py` as a `dict[str,
str]` and is auto-injected per row. **A category's negative prompt must
NOT exclude the target sound**: a hat category negative listing `"hi
hats"` would actively suppress the thing being requested. Always cross-
check when you add a new category.

---

## 6. `requirements.txt`

Create:

```txt
torch
torchaudio
torchsde
stable-audio-tools
einops
transformers
accelerate
safetensors
soundfile
numpy
pandas
tqdm
huggingface_hub
```

The shift from `diffusers` to `stable-audio-tools` is what lets us drive the
SA3 sampler ("pingpong"), pass `negative_conditioning` for loop/reverb
suppression, and access the underlying `sample_size` / `sample_rate` from
the model config.

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
mkdir -p /workspace/sas-sample-generator
cd /workspace/sas-sample-generator
```

Either copy your local project to the pod (from your Mac):

```bash
scp -P YOUR_RUNPOD_PORT -r ./sas-sample-generator root@YOUR_RUNPOD_HOST:/workspace/
```

Or `git clone` the repo directly on the pod — that's faster if it's pushed to GitHub.

---

## 7.4 Persistent volume layout (resume-after-kill)

When you terminate a pod, the pod's local disk is wiped. To survive that, RunPod
mounts a **Network Volume** at `/workspace` (you set this up in §7.1). Anything
that should outlive a pod termination MUST live under `/workspace`.

Recommended layout on the volume:

```text
/workspace/
  sas-sample-generator/       # git checkout of this repo
  .venv/                      # python env (reused across pods)
  .cache/huggingface/         # model weights (reused across pods)
  outputs/
    raw/                      # generated WAVs
    processed/
    rejected/
    manifests/
  .bash_env                   # env vars that point HF + outputs at the volume
```

Without this layout, every fresh pod re-runs `pip install` (~5 min) and
re-downloads Stable Audio 3 weights from Hugging Face (~5–8 GB for medium,
~2 GB for small, another ~2–3 min). With this layout, a new pod is ready in
~30 seconds because the venv and the model cache are already on the volume.

`scripts/setup.sh` (in this repo) sets all of this up. It is idempotent —
safe to re-run on every pod boot.

```bash
cd /workspace/sas-sample-generator
./scripts/setup.sh
source /workspace/.venv/bin/activate
hf auth login    # one time per volume
```

After this, `outputs/` for batch_generate.py is `/workspace/outputs` (set by
`SAS_OUTPUTS_DIR` in `.bash_env`). The `--skip-existing` flag in
`batch_generate.py` then lets a re-run continue where a killed run left off.

---

## 8. Python environment setup on the pod

The recommended path is `scripts/setup.sh` (see §7.4) — it creates the venv on
the persistent volume, installs the CUDA-matching PyTorch wheel, installs
`requirements.txt`, and verifies CUDA. Everything below is what the script does
manually, kept for reference / debugging:

```bash
cd /workspace/sas-sample-generator
python -m venv /workspace/.venv
source /workspace/.venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected output:

```text
CUDA available: True
Device: NVIDIA L40S
```

---

## 9. Hugging Face login

On your Mac or in a browser:

1. Go to Hugging Face.
2. Create an access token.
3. Accept the Stable Audio 3 model terms (and the Gemma terms) on the model page.

On the pod:

```bash
hf auth login
```

Paste your token.

---

## 10. Batch generation script

The working implementation lives at
[`scripts/batch_generate.py`](scripts/batch_generate.py) — read it there
rather than from a snapshot in this guide. The shape:

- Loads the SA3 model via `stable_audio_tools.get_pretrained_model(model_id)`
  in `torch.float16` on CUDA.
- Iterates one or more JSONL prompt files; supports multi-category batch
  generation in a single pipeline-load (only pay the model-load cost once).
- For each prompt, calls
  `stable_audio_tools.inference.generation.generate_diffusion_cond` with
  `conditioning=[{"prompt": ..., "seconds_total": duration}]` and a parallel
  `negative_conditioning` list to suppress loops / reverb / vocals.
- Defaults: `--steps 8`, `--cfg-scale 1.0`, `--sampler pingpong` — these
  are the values the SA3 model card specifies, and they're ~15× cheaper to
  evaluate than SAO 1.0's 120 / 7.0 defaults.
- Writes 24-bit PCM WAVs and a sibling `_metadata/{id}.json` per output;
  honours `--skip-existing` for idempotent re-runs.

---

## 11. Post-processing script

The working implementation lives at
[`scripts/postprocess_oneshots.py`](scripts/postprocess_oneshots.py) — read
it there rather than from a snapshot in this guide. The shape:

- Reads raw WAVs from `outputs/raw/<category>/` (or `--in-dir`),
  trims leading/trailing silence by RMS threshold (`--trim-threshold-db`,
  default -45 dB), truncates to `--max-seconds`, optionally downmixes
  to mono (`--mono`).
- Normalizes loudness via one of three modes (`--normalize`):
  - `lufs` (default): BS.1770-4 perceived loudness to `--target-lufs`
    (default -16 LUFS), with a hard `--peak-ceiling-db` (default
    -1 dBFS) applied after the LUFS gain. Loop-pads samples shorter
    than 3.5 s before the integrated-loudness measurement so very
    short hats/clicks still resolve.
  - `peak`: legacy peak-only normalize to `--target-peak-db`.
  - `none`: skip the gain stage entirely (still trims + fades).
- Writes 24-bit PCM WAVs with a sibling `<id>.txt` containing the
  generation prompt (read from `outputs/raw/<cat>/_metadata/<id>.json`
  written by `batch_generate.py`); also embeds the prompt in the WAV's
  RIFF INFO `ICMT` chunk via the `soundfile.SoundFile` context manager.
- Rejected samples (`too_quiet`, `silence`, `lufs_unmeasurable`,
  `normalization_failed`) go to `outputs/rejected/<category>/` with
  the same `.txt` sibling pattern, so you can grep which prompts
  produced bad output.

The previously-inline snapshot (pre-LUFS, pre-multi-category, pre-`.txt`
sidecars) has been removed in favor of pointing at the live file — the
implementation diverged enough that pinning a snapshot here misled more
than it helped.

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
  --out-root outputs/raw \
  --steps 8 \
  --cfg-scale 1.0 \
  --skip-existing
```

Post-process:

```bash
python scripts/postprocess_oneshots.py \
  --in-dir outputs/raw \
  --out-dir outputs/processed \
  --rejected-dir outputs/rejected \
  --mono \
  --max-seconds 2.5
```

Zip and download:

```bash
tar czf smoke_test_outputs.tar.gz outputs/processed
```

From your Mac:

```bash
scp -P YOUR_RUNPOD_PORT root@YOUR_RUNPOD_HOST:/workspace/sas-sample-generator/smoke_test_outputs.tar.gz .
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
  --out-root outputs/raw \
  --steps 8 \
  --cfg-scale 1.0 \
  --default-duration 1.5 \
  --skip-existing
```

Post-process:

```bash
python scripts/postprocess_oneshots.py \
  --in-dir outputs/raw \
  --out-dir outputs/processed \
  --rejected-dir outputs/rejected \
  --mono \
  --max-seconds 2.5
```

Create final ZIP:

```bash
tar czf stable_audio_kicks_1000.tar.gz outputs/processed
```

### 14.1 Get the data off the pod

For small runs (< 5 GB), `scp` the zip directly to your Mac:

```bash
scp -P YOUR_RUNPOD_PORT root@YOUR_RUNPOD_HOST:/workspace/sas-sample-generator/stable_audio_kicks_1000.tar.gz .
```

For larger / repeated runs, push to a cheap object store (Backblaze B2 ≈
$6/TB/month, Cloudflare R2 ≈ $15/TB/month with zero egress). One-time pod setup:

```bash
apt-get install -y rclone
rclone config           # create a remote named "samples" pointing at B2/R2
```

Then use `scripts/sync.sh`:

```bash
./scripts/sync.sh push              # push entire outputs/ tree
./scripts/sync.sh push outputs/processed   # push just the curated set
./scripts/sync.sh ls                # see what's on the remote
```

On your Mac, install rclone (`brew install rclone`), configure the same remote,
then `./scripts/sync.sh pull` to fetch.

The advantage of the object-store path: data durability is decoupled from
"is the pod running?" — you can terminate the pod, sleep on it, spin up a new
pod next week, and `sync.sh pull` to pick up where you left off.

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

Start with SA3's documented defaults:

```text
duration: 1.25–2.0 seconds
steps: 8       (SA3 default — converges much faster than SAO 1.0)
cfg-scale: 1.0 (SA3 default; higher values often degrade quality on SA3)
sampler: pingpong
num-waveforms-per-prompt: 1
```

Then experiment carefully — SA3's compute scaling is different from SAO 1.0:

```text
steps: 4–6     for cheap iteration on prompt wording
steps: 12–16   if 8 steps under-resolves transients on a specific category
cfg-scale: 1.5–3.0   if positive prompt isn't being respected enough
duration: 2.5 seconds for long 808 tails
num-waveforms-per-prompt: 2–3 for curation runs
```

Unlike SAO 1.0, blindly increasing steps on SA3 rarely pays for itself.
Tune the prompt first, the step count second.

### Loudness normalization

`postprocess_oneshots.py` perceived-loudness-normalizes every sample to
**-16 LUFS** (BS.1770-4) by default, with a hard **-1 dBFS** peak ceiling
applied afterwards. Samples shorter than ~3.5 s are loop-padded internally
just for the measurement — the gain is applied to the original buffer.

Three modes are available via `--normalize`:

```text
--normalize lufs   (default)  target -16 LUFS, peak ceiling -1 dBFS
--normalize peak              legacy mode: peak-only to --target-peak-db
--normalize none              skip the gain stage entirely
```

Tune the LUFS target if your sampler needs hotter signal:

```text
--target-lufs -14   streaming-hot (Spotify-ish)
--target-lufs -10   commercial-library hot (Splice-style)
```

The post-process per-sample measurements (LUFS in/out, peak ceiling
applied, etc.) are no longer shipped in a CSV manifest — they live only
in the pipeline run log (`/workspace/run.log` if you `tee`d it) and can
be re-derived from the processed WAVs at any time with `pyloudnorm`.
If you need an audit of how the empirical mean tracks `--target-lufs`,
run a quick `find outputs/processed -name '*.wav' | xargs -I{} python
-c 'import soundfile,pyloudnorm,sys;y,sr=soundfile.read(sys.argv[1]);
print(sys.argv[1], pyloudnorm.Meter(sr).integrated_loudness(y))' {}`
and adjust `--target-lufs` on the next run.

### Per-sample provenance (prompts travel with the WAVs)

Each processed WAV ships with its generation prompt attached in **two
places** so the data survives both the tar → scp boundary AND any
downstream "I dragged a WAV into Logic" detour:

1. **Sibling `<id>.txt`** in the same category folder — plain UTF-8
   text containing the positive prompt verbatim. Sits next to
   `<id>.wav` so merging two generation runs is a single `rsync`
   (no manifest CSV to reconcile, no `_metadata/` subdir to chase).
2. **Embedded WAV RIFF INFO chunks** — the positive prompt is also
   written to the `ICMT` (comment) chunk and the generator name to
   the `ISFT` (software) chunk. Logic, Ableton Live, Reaper,
   Audacity, `ffprobe`, and macOS Finder's Get-Info → "More Info"
   all surface these. So WAVs dragged outside the SAS workflow still
   carry the prompt.

Rejected samples get the same `<id>.txt` sibling under
`outputs/rejected/<cat>/`, so you can `grep -r '<keyword>' outputs/rejected`
to find which prompts produced silent or unmeasurable output.

Reading the WAV-embedded prompt from Python:

```python
import soundfile as sf
with sf.SoundFile("outputs/processed/kick/kick-c1da23da.wav") as f:
    print(f.comment)   # the original generation prompt
    print(f.software)  # "sas-sample-generator + Stable Audio 3 (libsndfile-...)"
```

Or just `cat outputs/processed/kick/kick-c1da23da.txt`.

**What's NOT in the shipped tar**: structured generation telemetry
(seed, model, exact sampler, steps, cfg-scale, per-sample LUFS in/out
values) lives only at `outputs/raw/<cat>/_metadata/<id>.json` on the pod
— `batch_generate.py` writes it pre-postprocess and the user's tar
boundary excludes `raw/` to keep downloads small. This is a deliberate
trade-off in favor of merge-friendly downstream layout. If you need
audit data on the Mac side, either grab `outputs/raw/<cat>/_metadata/`
separately, or re-measure LUFS with `pyloudnorm` after the fact.

This is **Phase 1** of a description-aware sample-selection feature
for the wider Signals & Sorcery ecosystem — Phase 2 will teach
`sas-drum-plugin`'s kit resolver to consult these prompts when ranking
candidates against the user's free-text request (so "funky 1960's
motown feel kick" preferentially picks the kick whose generation
prompt was closest in meaning). Phase 2 can read either the `.txt`
sibling or the WAV's `ICMT` chunk — both yield identical content.

Caveat: hand-edits to the processed `<id>.txt` are lost on the next
postprocess run. For stable annotations, edit the upstream raw
`_metadata/<id>.json` (the script only reads, never writes raw).

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
7. **Use 8 steps first (SA3 default), then only increase if quality requires it.**

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
hf auth login
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

## 23. When to switch to the custom Docker image

This repo includes a `Dockerfile`. It's optional. Decision rule:

| Situation | Use |
|-----------|-----|
| Iterating on prompts, < 1 pod boot per day | Stock RunPod template + `scripts/setup.sh` |
| Spinning up pods frequently and tired of waiting 5 min for `pip install` | Custom image |
| Sharing the setup with a teammate / running on a different cloud | Custom image |

Build and push once, then deploy:

```bash
docker build -t YOUR_DOCKERHUB_USER/sas-sample-generator:latest .
docker push YOUR_DOCKERHUB_USER/sas-sample-generator:latest
```

In RunPod, pick "Deploy a custom image" and paste the tag. The image deliberately
does NOT bake in the Stable Audio 3 weights — those still download into the
persistent volume's HF cache on first run, same as the stock-template flow. That
keeps the image small (~6 GB instead of ~10 GB) and avoids re-pushing on every
weight update.

---

## References

- Stable Audio 3 collection: https://huggingface.co/collections/stabilityai/stable-audio-3
- Stable Audio 3 Medium model card: https://huggingface.co/stabilityai/stable-audio-3-medium
- Stable Audio 3 Small SFX model card: https://huggingface.co/stabilityai/stable-audio-3-small-sfx
- Stable Audio Open paper (prior gen, for sampler / architecture background): https://arxiv.org/abs/2407.14358
- Stable Audio Tools (inference library this repo uses): https://github.com/Stability-AI/stable-audio-tools
- RunPod GPU pricing: https://www.runpod.io/pricing
- RunPod L40S page: https://www.runpod.io/gpu-models/l40s
- RunPod RTX 4090 page: https://www.runpod.io/gpu-models/rtx-4090
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
