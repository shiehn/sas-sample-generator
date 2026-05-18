# sas-sample-generator

Generate large batches of one-shot audio samples (kicks, snares, hats, etc.)
with [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
on a rented [RunPod](https://www.runpod.io) GPU. Prompts in (JSONL) → WAVs out.

Workflow assumes an **Apple Silicon Mac** as the control machine and a
**RunPod L40S 48GB pod** as the temporary GPU. The pod is ephemeral; a Network
Volume holds the venv, model weights, and generated audio so you can terminate
and resume freely.

For the long-form rationale (why these settings, prompt design, cost math, etc.),
see [`stable_audio_open_batch_oneshot_guide.md`](stable_audio_open_batch_oneshot_guide.md).
This README is the operational quickstart.

---

## Prerequisites

### Accounts

- **Hugging Face** — create an account, then visit the
  [Stable Audio Open model page](https://huggingface.co/stabilityai/stable-audio-open-1.0)
  and accept the license. Generate a read-only access token under Settings →
  Access Tokens. You'll paste this on the pod via `huggingface-cli login`.
- **RunPod** — create an account and add a payment method.
- **Object storage (optional but recommended for large runs)** — Backblaze B2
  (~$6/TB/month) or Cloudflare R2 (~$15/TB/month, zero egress). Skip this for
  small smoke tests; use `scp` instead.

### Local Mac tools

```bash
brew install rclone     # only needed if you'll push outputs to B2/R2
# scp and ssh ship with macOS
```

Nothing else is required on the Mac for the basic flow. Python runs on the pod.

---

## One-time RunPod setup

### 1. Create a Network Volume

In the RunPod console: **Storage → Network Volumes → New Volume**.

| Field | Value |
|---|---|
| Region | pick one with L40S availability (e.g. `US-OR-1`) |
| Size | **100 GB** (raise later if you blow past it) |
| Name | `sas-samples` |

The volume is billed monthly per GB and survives pod terminations. This is what
makes resume-after-kill work.

### 2. Deploy a pod attached to the volume

**Pods → Deploy → GPU Pod**:

| Field | Value |
|---|---|
| GPU | **L40S 48GB** (or RTX 4090 24GB for cheaper smoke tests) |
| Region | same region as the volume |
| Network Volume | select `sas-samples`, mount path `/workspace` |
| Template | `RunPod PyTorch 2.x` (any recent CUDA 12.x build) |
| Container Disk | default (~20 GB is plenty; the volume holds the heavy stuff) |
| Expose | SSH (port 22) |

Click **Deploy**. Wait ~30 seconds for the pod to come up, then copy the SSH
command from the pod's "Connect" panel — it looks like:

```text
ssh root@123.45.67.89 -p 12345 -i ~/.ssh/id_ed25519
```

---

## Bootstrap the pod (first time on this volume)

From your Mac terminal:

```bash
ssh root@YOUR_RUNPOD_HOST -p YOUR_RUNPOD_PORT
```

On the pod:

```bash
cd /workspace
git clone https://github.com/YOUR_GITHUB_USER/sas-sample-generator.git
cd sas-sample-generator

# Creates /workspace/.venv, installs CUDA torch + requirements, points HF cache
# at /workspace/.cache/huggingface. Idempotent — safe to re-run on every boot.
./scripts/setup.sh

source /workspace/.venv/bin/activate
huggingface-cli login    # paste your HF token; stored under /workspace/.cache
```

Expected output from `setup.sh` at the end:

```text
[setup] cuda available: True
[setup] device:         NVIDIA L40S
[setup] done. ...
```

If `cuda available: False`, you booted onto a CPU template by mistake — destroy
the pod and redeploy with a GPU pod template.

---

## Smoke test (3 samples, ~30 seconds of GPU time)

A `prompts/kicks_smoke_test.jsonl` is already in the repo. From the pod, in the
project directory with the venv active:

```bash
python scripts/batch_generate.py \
  --prompts prompts/kicks_smoke_test.jsonl \
  --out /workspace/outputs/raw \
  --steps 80 \
  --skip-existing

python scripts/postprocess_oneshots.py \
  --in-dir /workspace/outputs/raw \
  --out-dir /workspace/outputs/processed \
  --rejected-dir /workspace/outputs/rejected \
  --manifest /workspace/outputs/manifests/smoke_test_manifest.csv \
  --mono \
  --max-seconds 2.5
```

You should end up with 3 WAVs in `/workspace/outputs/processed/`. Listen to one
on the pod via `head -c 200000 /workspace/outputs/processed/kick_test_0001.wav`
through a player, or skip ahead to the "Get the data off the pod" step.

---

## Full batch run

### Build the prompts file

Write descriptions in a plain text file, **one per line** — much easier than
hand-writing 1000 lines of JSONL. Blank lines and lines starting with `#` are
ignored. Example `prompts/kicks.txt`:

```text
# 909-style
tight 909-style kick drum one shot, hard transient, short decay, clean low end, no melody, no loop
punchy 909 kick drum one shot, sharp click, short body, dry, no hi hats, no snare, no cymbals

# 808-style
deep 808-style kick drum one shot, long sub bass decay, smooth sine low end, clean transient, dry
warm 808 kick drum one shot, saturated low end, medium decay, dry, no vocals, no melody

# Techno
deep techno kick drum one shot, warm saturated low end, short click transient, mono-compatible, dry
distorted industrial techno kick drum one shot, aggressive transient, saturated body, no cymbals
```

Convert it to JSONL with `scripts/list_to_jsonl.py`:

```bash
# From your Mac (or on the pod, doesn't matter — pure stdlib Python)
python3 scripts/list_to_jsonl.py \
  --in prompts/kicks.txt \
  --out prompts/kicks.jsonl \
  --prefix kick \
  --duration 1.5
```

Produces:

```jsonl
{"id": "kick_0001", "prompt": "tight 909-style kick drum one shot, ...", "seed": 1001, "duration": 1.5}
{"id": "kick_0002", "prompt": "punchy 909 kick drum one shot, ...", "seed": 1002, "duration": 1.5}
...
```

Flags worth knowing:

| Flag | Default | What it does |
|---|---|---|
| `--prefix` | input filename stem | ID prefix; override to get `kick_0001` from `kicks.txt` |
| `--start-seed` | `1001` | First seed; subsequent rows increment by 1 |
| `--start-id` | `1` | First ID number |
| `--duration` | `1.5` | Seconds per sample (applies to every row) |
| `--pad` | `4` | Zero-pad width for IDs (`0001` vs `001`) |

The helper dedupes identical lines, normalizes whitespace, and warns about
skipped rows to stderr. If you need per-row duration overrides (e.g. 808s with
longer tails), edit the resulting JSONL or maintain separate `.txt` files per
duration target.

### Copy the JSONL to the pod and run

```bash
# From your Mac
scp -P YOUR_RUNPOD_PORT prompts/kicks.jsonl \
  root@YOUR_RUNPOD_HOST:/workspace/sas-sample-generator/prompts/
```

Run on the pod (inside the venv):

```bash
python scripts/batch_generate.py \
  --prompts prompts/kicks.jsonl \
  --out /workspace/outputs/raw \
  --steps 120 \
  --cfg-scale 7 \
  --default-duration 1.5 \
  --skip-existing

python scripts/postprocess_oneshots.py \
  --in-dir /workspace/outputs/raw \
  --out-dir /workspace/outputs/processed \
  --rejected-dir /workspace/outputs/rejected \
  --manifest /workspace/outputs/manifests/kicks_manifest.csv \
  --mono \
  --max-seconds 2.5
```

If the run dies partway through, just re-run — `--skip-existing` resumes from
the last successful WAV. The outputs are on the persistent volume, so even a
full pod termination is recoverable.

---

## Get the data off the pod

### Small runs (≤ 5 GB): zip + scp

On the pod:

```bash
cd /workspace
zip -r kicks_1000.zip outputs/processed outputs/manifests outputs/raw/_metadata
```

On your Mac:

```bash
scp -P YOUR_RUNPOD_PORT \
  root@YOUR_RUNPOD_HOST:/workspace/kicks_1000.zip ~/Downloads/
```

### Large / repeated runs: rclone to B2 or R2

One-time setup on the pod:

```bash
apt-get install -y rclone
rclone config
```

`rclone config` is interactive. For Backblaze B2, pick `b2`, paste your
Application Key ID and Application Key, name the remote `samples`. For
Cloudflare R2, pick `s3` → `Cloudflare R2` and follow the prompts.

Test it:

```bash
rclone lsd samples:
```

Then use the wrapper:

```bash
./scripts/sync.sh push                     # push outputs/ to samples:sas-samples/<hostname>/
./scripts/sync.sh push outputs/processed   # only push the curated tree
./scripts/sync.sh ls                       # see what's on the remote
./scripts/sync.sh size                     # total bytes on the remote
```

On your Mac, set up the same rclone remote and pull:

```bash
brew install rclone
rclone config                              # mirror the pod's "samples" remote
cd ~/path/to/sas-sample-generator
RUN_TAG=<the-pod-hostname-from-the-push> ./scripts/sync.sh pull
```

`RUN_TAG` defaults to the current machine's hostname on both push and pull. On
the Mac you have to set it explicitly to the pod's hostname so the pull lands
on the right path.

---

## Stop / resume

When a run is done:

1. Confirm the data is somewhere durable — either on the Network Volume (it
   stays there for as long as the volume exists) or pushed to B2/R2.
2. **Stop or terminate the pod from the RunPod console.** GPU billing stops
   immediately; the volume keeps billing (~$0.07/GB/month).

Coming back later:

1. Deploy a new pod, attach the same Network Volume.
2. SSH in. The repo, venv, HF cache, and outputs are all still under
   `/workspace/`. Just re-activate the venv:

   ```bash
   source /workspace/.venv/bin/activate
   cd /workspace/sas-sample-generator
   ```

3. Re-run `./scripts/setup.sh` if you want — it's idempotent and will pick up
   any new requirements you've added since the last boot.

What does NOT survive a pod termination:
- `/root/.bashrc` exports (re-run by `setup.sh`)
- Anything written outside `/workspace/`
- The pod's SSH host key (your Mac will warn about a changed key; that's
  expected — clear it with `ssh-keygen -R '[host]:port'`)

---

## Local prep on macOS

Day-to-day, you'll be:

1. **Writing prompts** in `prompts/*.jsonl`. JSONL = one JSON object per line;
   field reference is in
   [`stable_audio_open_batch_oneshot_guide.md` §5](stable_audio_open_batch_oneshot_guide.md).
2. **Auditioning samples** after `sync.sh pull` or `scp`. Drag the unzipped
   `outputs/processed/` into your DAW of choice or QuickLook them in Finder.
3. **Curating**. The model overproduces; throw out the bad ones manually. The
   manifest CSV (`outputs/manifests/*.csv`) tells you what was auto-rejected
   and why.

`outputs/` is gitignored — that's deliberate, you don't want gigs of WAVs in
git.

---

## Repo layout

```text
sas-sample-generator/
├── README.md                                 # you are here
├── stable_audio_open_batch_oneshot_guide.md  # long-form reference
├── Dockerfile                                # optional custom image (see below)
├── .dockerignore
├── requirements.txt
├── prompts/
│   └── kicks_smoke_test.jsonl                # ships with the repo
├── scripts/
│   ├── setup.sh                              # idempotent pod bootstrap
│   ├── sync.sh                               # rclone wrapper for B2/R2
│   ├── list_to_jsonl.py                      # text list -> JSONL helper
│   ├── batch_generate.py                     # the actual generator
│   ├── postprocess_oneshots.py               # trim / normalize / reject
│   └── benchmark.py                          # per-sample timing + cost estimate
└── outputs/                                  # gitignored; generated WAVs land here
    ├── raw/
    ├── processed/
    ├── rejected/
    └── manifests/
```

---

## Optional: custom Docker image

Stock RunPod templates + `setup.sh` add ~5 minutes to a cold start (pip
install, torch download). If you're booting pods often and that bothers you,
build a custom image:

```bash
docker build -t YOUR_DOCKERHUB_USER/sas-sample-generator:latest .
docker push YOUR_DOCKERHUB_USER/sas-sample-generator:latest
```

In RunPod, deploy with **Custom Image** and paste the tag. Cold start drops to
~30s. Model weights still download into the Network Volume on first run — the
image deliberately doesn't bake them in.

Skip this until you're past prototyping. It's not worth the iteration loop
overhead during prompt design.

---

## Security notes

This is a public repo. A few rules:

- **Never commit secrets.** `.env`, `*.token`, `*.secret`, `.huggingface/` are
  all in `.gitignore`. Hugging Face tokens, RunPod API keys, B2/R2
  Application Keys, and SSH private keys live only on disk + in the pod's
  environment.
- **Prompts can be public.** They don't contain anything sensitive.
- **Generated WAVs are gitignored.** If you want to publish a sample pack,
  push it to a Hugging Face dataset or release it via your own channel.

---

## Cost guardrails

- **Stop the pod when you walk away.** GPU billing is per-second; an L40S pod
  left running overnight is ~$20 of nothing.
- **Smoke-test at low step counts** (`--steps 80`) before committing to a long
  batch at 120+.
- **Use `--skip-existing`.** A killed run resumes for free.
- **Keep the Network Volume sized to actual need.** 100 GB ≈ $7/month.
  Generated WAVs are ~400 KB each at 1.5s stereo 24-bit, so 100 GB holds
  ~250k one-shots.

Per-sample cost math is in
[`stable_audio_open_batch_oneshot_guide.md` §19](stable_audio_open_batch_oneshot_guide.md).
