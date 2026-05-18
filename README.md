# sas-sample-generator

<p align="center">
  <img src="docs/img/cauldron.png" alt="sas-sample-generator" width="360" />
</p>

Generate batches of one-shot audio samples (kicks, snares, hats, etc.) with
[Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
on a rented [RunPod](https://www.runpod.io) GPU.

**Designed for occasional use.** This README is the recipe — read top to bottom,
copy-paste each command block, finish in ~30 minutes for ~$0.30 of GPU time.

Assumes an **Apple Silicon Mac** as the control machine.

For the rationale (why these settings, prompt-design tips, deep cost math),
see [`stable_audio_open_batch_oneshot_guide.md`](stable_audio_open_batch_oneshot_guide.md).

> Part of the [Signals & Sorcery](https://signalsandsorcery.com) family.
> See [Related repos](#signals--sorcery-family) at the bottom of this README.

---

## What you'll provide

A plain text file with **one description per line** for each drum/percussion
category you want. The repo ships with **14 starter categories** (1400+
prompts total) — you can run them as-is, edit them, or subset which to
generate.

Each line in a `prompts/<category>.txt` becomes one generated WAV. Example
from [`prompts/kick.txt`](prompts/kick.txt):

```text
# 909-style
tight 909-style kick drum one shot, hard click transient, short punchy body, dry
punchy 909 kick drum one shot, sharp transient, controlled low end, clean studio sample

# 808-style
deep 808 kick one shot, long sub bass decay, smooth sine low end, dry
warm 808 kick one shot, saturated low end, medium decay, dry, no melody, no loop
```

Blank lines and lines starting with `#` are ignored (handy for grouping).
Aim for ~10 words per line. Always include phrases like `one shot, no loop`
so the model doesn't render a rhythmic loop.

### Categories shipped with the repo

| Category | Duration | Prompts |
|---|---|---|
| `kick` | 1.5s | 102 |
| `snare-standard` | 1.0s | 101 |
| `snare-rim` | 0.75s | 100 |
| `hat-closed` | 0.5s | 103 |
| `hat-open` | 1.5s | 101 |
| `cymbal-ride` | 2.5s | 100 |
| `cymbal-crash` | 3.0s | 100 |
| `cymbal-splash` | 1.5s | 100 |
| `tamborine` | 1.0s | 101 |
| `shaker` | 0.75s | 102 |
| `tom-hi` | 1.0s | 100 |
| `tom-mid` | 1.25s | 100 |
| `tom-low` | 1.5s | 100 |
| `hit` | 1.5s | 102 |
| **Total** |  | **1412** |

Output filenames are content-addressed: `{category}-{8-char-hash}.wav`. Same
prompt + seed → same filename → safely re-runnable with `--skip-existing`.

To **subset** what gets generated, edit
[`scripts/categories.txt`](scripts/categories.txt) — comment out any line to
skip that category.

---

## ONE-TIME SETUP (do once, then forget)

### A. Hugging Face

1. Create / sign in at [huggingface.co](https://huggingface.co).
2. Visit [stabilityai/stable-audio-open-1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
   and click **Agree and access repository**.
3. Create a read-only access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
   Save it in your password manager — you'll paste it once per pod.

### B. RunPod

1. Create / sign in at [runpod.io](https://runpod.io). Add a payment method.
2. Add your Mac SSH public key under
   [Settings → SSH Keys](https://www.runpod.io/console/user/settings):
   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub        # copies key to clipboard
   ```
   Paste it into the form. (If `~/.ssh/id_ed25519.pub` doesn't exist:
   `ssh-keygen -t ed25519` first, accept defaults.)

---

## EVERY-RUN STEPS

### 1. Deploy a pod

[runpod.io/console/pods](https://www.runpod.io/console/pods) → **Deploy → GPU Pod**:

| Setting | Value |
|---|---|
| GPU | **RTX A6000** (48 GB VRAM, ~$0.49/hr) |
| Template | most recent **RunPod PyTorch** with CUDA 12.x |
| Container Disk | 50 GB (default) |
| Volume Disk | 100 GB at `/workspace` |
| Expose | SSH (port 22) — default |

Click **Deploy On-Demand**. Wait ~30 sec until status is `RUNNING`.

On the pod's card click **Connect → SSH over exposed TCP** and copy the SSH
command. It looks like:

```text
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

### 2. SSH into the pod

From your Mac terminal, paste the SSH command from step 1. Type `yes` to
accept the host key on first connect.

If you get `Permission denied (publickey)`:
```bash
ssh-add ~/.ssh/id_ed25519
```
…then retry.

### 3. Clone + bootstrap (~4–5 min)

On the pod:
```bash
cd /workspace && \
git clone https://github.com/shiehn/sas-sample-generator.git && \
cd /workspace/sas-sample-generator && \
./scripts/setup.sh 2>&1 | tee /workspace/setup.log
```

**Why these paths matter** (and the reason this used to be slow): `/workspace`
is a network filesystem (MooseFS) — fine for big sequential reads/writes
(model weights, generated audio) but painfully slow for many-tiny-files (a
Python venv). The script installs the venv at **`/root/.venv`**, which is on
the pod's container-local SSD, and only keeps the HuggingFace cache and
outputs on `/workspace`. Roughly:

```text
/root/.venv                   ← Python venv          (fast SSD; ~5 min install)
/workspace/sas-sample-generator   ← cloned repo
/workspace/.cache/huggingface ← model weights        (downloaded once)
/workspace/outputs            ← generated WAVs
```

You're done when you see:
```text
[setup] cuda available: True
[setup] device:         NVIDIA RTX A6000
[setup] done.
[setup] next: source /root/.venv/bin/activate
```

### 4. Hugging Face login

```bash
source /root/.venv/bin/activate
hf auth login
```

Paste your HF token (One-Time Setup A). Answer `n` to "Add token as git
credential".

### 5. (Optional) Edit prompts and choose categories

The 14 prompt files are already in `prompts/<category>.txt`. To run them
as-is, **skip to step 6**.

To customize:

- **Edit content**: `nano prompts/kick.txt` (or scp over your own version,
  or edit on Mac → `git push` → `git pull` on the pod).
- **Subset which categories run**: edit
  [`scripts/categories.txt`](scripts/categories.txt) and comment out the
  lines you want to skip. Useful for prompt iteration on a single category.

### 6. Run the whole pipeline (~5 hours for all 14, ~15 min for one)

Wrap the run in `tmux` first so an SSH drop doesn't kill the job:

```bash
tmux new -s sas
./scripts/run_all.sh 2>&1 | tee /workspace/run.log
```

(Detach with `Ctrl-b d`; reattach later with `tmux attach -t sas`.)

The wrapper does three things in order:

1. For each enabled category, run `list_to_jsonl.py` to build
   `prompts/<cat>.jsonl`.
2. Run `batch_generate.py` **once** with all the JSONL paths. The model
   pipeline loads only once — looping per-category would re-load it 14
   times and waste ~30 minutes.
3. Run `postprocess_oneshots.py --category <cat>` for each category in
   turn (trim, normalize, mono).

First call downloads Stable Audio Open (~3–5 GB, ~3 min, one-time). Then
~10 sec/sample × prompt count.

**Single-category iteration** (when you're tuning prompts):

```bash
# 1. Edit prompts/kick.txt
nano prompts/kick.txt

# 2. Comment out everything except `kick` in scripts/categories.txt
# 3. Re-run — --skip-existing means you only generate the new/changed prompts
./scripts/run_all.sh
```

### 7. (skipped — folded into step 6)

### 8. Verify the outputs

```bash
ls /workspace/outputs/processed/
# Should show 14 subdirs, each with ~100 .wav files
find /workspace/outputs/processed -name "*.wav" | wc -l
# Should show ~1400 (minus any auto-rejected silent samples)
```

Per-category manifests with CSV stats live at
`/workspace/outputs/manifests/<category>.csv`.

### 9. Zip and download

On the pod:
```bash
cd /workspace
tar czf run.tar.gz outputs/processed outputs/manifests
ls -lh run.tar.gz
```

(We use `tar` rather than `zip` because the stock RunPod PyTorch image
doesn't ship `zip`. `tar` is preinstalled everywhere. `tar` also recurses
into the per-category subdirs automatically.)

In a **second** Mac terminal (don't close the SSH session yet — you still
need it for step 10):

```bash
cd ~/Downloads
scp -P <POD_PORT> root@<POD_IP>:/workspace/run.tar.gz .
tar xzf run.tar.gz
open outputs/processed                 # Finder + QuickLook to audition
```

`<POD_PORT>` and `<POD_IP>` are the same ones from your step-1 SSH command.
The unpacked structure is one folder per category:

```text
outputs/processed/
  kick/        kick-c1da23da.wav   kick-e5d95885.wav   ...
  snare-standard/   snare-standard-...wav
  hat-closed/  ...
  ...etc
```

### 10. ⚠️ TERMINATE THE POD

This is the step you will forget. The pod bills **$0.49/hr** for as long as
it exists, whether you're using it or not.

- **Idle overnight** ≈ $12
- **Forgotten for a week** ≈ $80
- **Forgotten for a month** ≈ $350

In the [RunPod console](https://www.runpod.io/console/pods), click your pod's
card → **Terminate**. Confirm.

Termination wipes `/workspace`. That's fine — you have the zip on your Mac.
Next month, you start fresh from step 1.

---

## File layout

```text
sas-sample-generator/
├── README.md                                   ← you are here
├── stable_audio_open_batch_oneshot_guide.md    ← long-form background
├── requirements.txt
├── prompts/
│   ├── kick.txt          kick.jsonl
│   ├── snare-standard.txt    snare-standard.jsonl
│   ├── snare-rim.txt    snare-rim.jsonl
│   ├── hat-closed.txt   hat-closed.jsonl
│   ├── hat-open.txt     hat-open.jsonl
│   ├── cymbal-ride.txt  cymbal-ride.jsonl
│   ├── cymbal-crash.txt cymbal-crash.jsonl
│   ├── cymbal-splash.txt cymbal-splash.jsonl
│   ├── tamborine.txt    tamborine.jsonl
│   ├── shaker.txt       shaker.jsonl
│   ├── tom-hi.txt       tom-hi.jsonl
│   ├── tom-mid.txt      tom-mid.jsonl
│   ├── tom-low.txt      tom-low.jsonl
│   └── hit.txt          hit.jsonl
├── scripts/
│   ├── setup.sh                                ← step 3 — bootstrap the pod
│   ├── run_all.sh                              ← step 6 — full pipeline
│   ├── categories.txt                          ← which categories to include
│   ├── category_config.py                      ← per-category negatives + durations
│   ├── list_to_jsonl.py                        ← .txt → .jsonl converter (called by run_all)
│   ├── batch_generate.py                       ← GPU inference (called by run_all)
│   ├── postprocess_oneshots.py                 ← trim / normalize (called by run_all)
│   ├── benchmark.py                            ← optional: per-sample cost math
│   └── sync.sh                                 ← optional: rclone to B2 / R2
└── outputs/                                    ← gitignored; generated WAVs land here
    ├── raw/<category>/                         ← raw model output
    ├── processed/<category>/                   ← trimmed + normalized
    ├── rejected/<category>/                    ← samples auto-rejected for silence
    └── manifests/<category>.csv                ← per-category processing log
```

---

## When something breaks

| Symptom | Most likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` on ssh | private key not loaded into agent | `ssh-add ~/.ssh/id_ed25519` |
| `setup.sh` hangs at `Installing collected packages:` for >5 min | something redirected the venv onto `/workspace` (MooseFS); script defaults to `/root/.venv` for a reason | check `echo $VENV_DIR` — should be `/root/.venv`. If overridden, unset it and re-run |
| `cuda available: False` after `setup.sh` | picked a CPU template | terminate; re-deploy with PyTorch GPU template |
| `huggingface_hub.utils._errors.GatedRepoError` | didn't accept the SAO license | visit the [model page](https://huggingface.co/stabilityai/stable-audio-open-1.0), click "Agree" |
| `batch_generate.py` errors `CUDA out of memory` | duration too long for VRAM | lower `--default-duration` or `--num-waveforms-per-prompt 1` |
| All samples sound like loops | prompts not specific enough | add `one shot, no loop, no hi hats, no snare` to every prompt |
| Too much reverb | model adds ambience by default | add `dry, no reverb, no ambience` to prompts |
| Generated WAV doesn't sound like the target category (e.g. hats sound like snares) | the per-category `negative_prompt` may be excluding the target — bug in `scripts/category_config.py` | open `scripts/category_config.py`, audit the negative for that category; nothing in it should match the target sound |
| `run_all.sh` skips a category | corresponding `prompts/<cat>.txt` is missing or has only comments | check `prompts/` has the .txt file and contains non-comment lines |
| SSH disconnects mid-run | network blip + foregrounded run | use `tmux new -s sas` BEFORE running, reattach with `tmux attach -t sas` |

---

## Cost recap

On an RTX A6000 at $0.49/hr. Both numbers below assume a fresh pod (so they
include the one-time bootstrap + model download).

| Run shape | Time | Cost |
|---|---|---|
| **Single category** (~100 samples) | ~20 min | ~$0.16 |
| **All 14 categories** (~1400 samples) | ~5 hr | ~$2.45 |

The "all 14" cost is dominated by GPU inference time (~10 sec × 1400 = ~4 hr).
The bootstrap + model download is one-time per pod and shared across all
categories, which is why the per-sample cost stays roughly flat as you scale.

If you keep the pod alive between runs in the same session (e.g., iterating
prompts on one category), each subsequent run is just step 6 again with
`--skip-existing` skipping work you've already paid for.

---

## Security

This is a **public repo**. Never commit:
- Hugging Face tokens
- RunPod API keys
- B2 / R2 / S3 keys
- SSH private keys
- The generated WAVs (gitignored already)

`.gitignore` covers `.env`, `*.token`, `*.secret`, `outputs/*`. If you ever
`git add` a file containing a secret by mistake: **rotate the secret first**,
then `git rm` + commit + push. Treat anything that hit `main` as compromised.

---

## Long-form reference

[`stable_audio_open_batch_oneshot_guide.md`](stable_audio_open_batch_oneshot_guide.md)
covers:
- Why Stable Audio Open vs alternatives
- Prompt-design rules and category-specific templates
- Optional persistent Network Volume layout (for users running multiple times per week)
- Optional rclone push to Backblaze B2 / Cloudflare R2 instead of `scp`
- Optional custom Docker image
- Cost-control deep dive

---

## Signals & Sorcery family

This is one piece of a larger ecosystem around the
[Signals & Sorcery](https://signalsandsorcery.com) audio app.

**Plugin SDK & templates**
- [sas-plugin-sdk](https://github.com/shiehn/sas-plugin-sdk) — types, components, and hooks for building generator plugins
- [sas-plugin-template](https://github.com/shiehn/sas-plugin-template) — starter template for new plugins
- [sas-chat-plugin](https://github.com/shiehn/sas-chat-plugin) — in-app conversational agent

**Built-in plugins**
- [sas-audio-plugin](https://github.com/shiehn/sas-audio-plugin) — default AI audio generation plugin
- [sas-sample-plugin](https://github.com/shiehn/sas-sample-plugin) — default sample plugin
- [sas-synth-plugin](https://github.com/shiehn/sas-synth-plugin) — default synth plugin
- [sas-texture-plugin](https://github.com/shiehn/sas-texture-plugin) — texture/ambient plugin
- [sas-recorder-plugin](https://github.com/shiehn/sas-recorder-plugin) — line-in recording plugin

**Audio tooling**
- [sas-audio-processor](https://github.com/shiehn/sas-audio-processor) — audio processing utilities
- [Signals2Surge](https://github.com/shiehn/Signals2Surge) — synth patch transfer to Surge XT

**Infrastructure**
- [signals-and-sorcery-server](https://github.com/shiehn/signals-and-sorcery-server) — DAWNet API + WebSocket server
- [signals-and-sorcery-docs](https://github.com/shiehn/signals-and-sorcery-docs) — public docs

**Other**
- [signalsandsorcery-game-ui](https://github.com/shiehn/signalsandsorcery-game-ui) — LLM-powered RPG frontend
- [SignalsAndSorcery](https://github.com/shiehn/SignalsAndSorcery) — earlier VueJS + Web Audio sample arrangement tool
- [Errantry](https://github.com/shiehn/Errantry) — E2E testing for agent-facing CLIs (drives this project's CLI too)
