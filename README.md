# sas-sample-generator

<p align="center">
  <img src="docs/img/cauldron.png" alt="sas-sample-generator" width="360" />
</p>

Generate batches of one-shot audio samples (kicks, snares, hats, etc.) with
[Stable Audio 3](https://huggingface.co/stabilityai/stable-audio-3-medium)
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
2. Visit [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
   and click **Agree and access repository**. SA3 also requires accepting
   the Gemma Terms of Use (linked from the same page). If you switch models
   via `--model`, accept the license on that model's page too —
   [stable-audio-3-small-sfx](https://huggingface.co/stabilityai/stable-audio-3-small-sfx)
   is the lighter 0.6B SFX-tuned alternative.
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

### 6. Run the whole pipeline (~25 min for all 14, ~2 min for one)

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
   turn (trim, perceived-loudness normalize to **-16 LUFS** with a -1 dBFS
   peak ceiling, mono downmix). Each WAV ships with a sibling `<id>.txt`
   in the same category folder containing its exact generation prompt;
   the prompt is also embedded in the WAV's RIFF INFO comment chunk so
   Logic / Ableton / Audacity / `ffprobe` / macOS Get-Info can show it.
   Merging samples from multiple runs is a single `rsync` — no manifest
   to reconcile.

First call downloads Stable Audio 3 (~5–8 GB for medium, ~2 GB for small, ~3 min,
one-time). Then ~1 sec/sample × prompt count — SA3 converges in ~8 diffusion
steps (vs 120 for SAO 1.0), so generation is ~10–15× faster.

Outputs are LUFS-normalized by default so kicks, hats, and splashes all
sit at the same perceived volume in your sampler. To revert to the old
peak-only behavior pass `--normalize peak` to `postprocess_oneshots.py`,
or override the target with `--target-lufs -14` (streaming-hot) or
`--target-lufs -10` (commercial library hot).

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

Every `<id>.wav` ships with a sibling `<id>.txt` containing its
generation prompt. Spot-check the pairing:

```bash
test "$(find /workspace/outputs/processed -name '*.wav' | wc -l)" \
   = "$(find /workspace/outputs/processed -name '*.txt' | wc -l)" \
   && echo "wav/txt pairing OK"
```

### 9. Zip and download

On the pod:
```bash
cd /workspace
tar czf run.tar.gz outputs/processed
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
    ├── raw/<category>/<id>.wav                 ← raw model output
    ├── raw/<category>/_metadata/<id>.json      ← prompt + gen params (written by batch_generate.py; stays on pod)
    ├── processed/<category>/<id>.wav           ← trimmed + LUFS-normalized; ICMT chunk = prompt
    ├── processed/<category>/<id>.txt           ← sibling positive prompt (plain text)
    ├── rejected/<category>/<id>.wav            ← samples auto-rejected for silence / LUFS failure
    └── rejected/<category>/<id>.txt            ← prompt that produced the rejection
```

---

## Pitched-instrument pipeline (Phase 1.0+)

Sibling pipeline to the drum one above. Same Stable Audio 3 generator,
different downstream: adds quality gates (CREPE pitch detection,
BasicPitch polyphony when TF/numpy ABI is happy, librosa onset, custom
sustain-plateau), smart pitch correction (snap to nearest semitone or
shift to target — see "Smart pitch correction" below), and zone
pre-rendering (RubberBand R3 with formant preservation). Emits a
per-instrument `manifest.json` consumed by `sas-instrument-plugin`.

**16 categories ship with the repo, 1,500+ prompts total:**

| Category | Prompts | Target | Duration | Notes |
|---|---|---|---|---|
| basses | 95 | E2 (40) | 6.0s | 808 sub, reese, analog, acid, fuzz, electric |
| bells | 86 | C5 (72) | 4.0s | glockenspiel, FM/DX7, music box |
| brass | 85 | A3 (57) | 6.0s | analog 80s, trumpet/trombone, sax sections |
| fx | 96 | A4 (69) | 4.0s | risers, downlifters, impacts, atmospheres |
| guitars | 95 | E3 (52) | 4.0s | clean electric, nylon, acoustic, slide |
| keys | 87 | C3 (48) | 5.0s | Rhodes, Wurlitzer, clavinet, DX7 |
| mallets | 97 | C4 (60) | 3.0s | marimba, vibes, kalimba, steel drum |
| organs | 93 | C3 (48) | 8.0s | Hammond B3, pipe, combo, theatre |
| pads | 96 | C3 (48) | 12.0s | warm analog, ethereal, lo-fi, evolving |
| percussion | 91 | C4 (60) | 2.0s | pitched hits, tonal hand drums |
| pianos | 98 | C4 (60) | 5.0s | grand, upright, felt, jazz/soul, cinematic |
| plucks | 101 | C4 (60) | 3.0s | acoustic, electric, world, lo-fi, trap |
| strings | 104 | A3 (57) | 8.0s | section/solo, synth, cinematic, pizzicato |
| synths | 101 | C3 (48) | 5.0s | analog mono, supersaw, FM, wavetable, acid |
| vocals | 99 | A3 (57) | 5.0s | choir, vocal chops, vocoded, world |
| winds | 104 | A4 (69) | 5.0s | flute, sax, clarinet, shakuhachi |

To **subset** which categories run, edit
[`scripts/pitched_categories.txt`](scripts/pitched_categories.txt) —
comment any line with `#` to skip that category.

### Per-pod system prerequisites

`scripts/setup.sh` installs **everything** the pitched pipeline needs:

- All Python deps from `requirements.txt` (librosa, torchcrepe, basic-pitch, pyrubberband, soxr, …)
- All system packages — `rsync` (transfer to Mac), `tmux` (long sessions survive disconnect), `rubberband-cli` (pyrubberband backend), `ffmpeg` (audio inspection), `zip`/`unzip` (archives)
- The compiled stable-audio-tools from git main (the PyPI release doesn't support SA3-medium)
- CUDA-12.8 torch wheels (Blackwell-compatible, also works on older Hopper/Ampere/Ada)

You should **never** need to apt-get or pip install anything on a pod
after running `setup.sh`. If you do, treat it as a bug in `setup.sh` and
add it there.

### On your Mac (for the local enrich step)

```bash
brew install rubberband                          # pitch-shift backend
cd ~/path/to/sas-sample-generator                # your local repo
pip install -r requirements.txt                  # one-time, ~5 min
```

---

### EVERY-RUN STEPS (pitched)

Designed to be safely repeatable from a cold start. The whole pipeline:
**~15 min setup + ~30 min generate + ~10 min gate + ~30 min enrich + transfer**.

#### Step 1 — Deploy a fresh pod

[runpod.io/console/pods](https://www.runpod.io/console/pods) → **Deploy → GPU Pod**:

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX A6000 / 4090 / 5090 / L40S / A100** (24+ GB VRAM) | SA3-medium fits in 16 GB; 24 GB gives headroom |
| Template | most recent **RunPod PyTorch** with CUDA 12.x | matches our cu128 wheels |
| **Container Disk** | **100 GB** ⚠️ THE IMPORTANT ONE | persistent across pod restart; holds venv + HF model cache + outputs |
| **Network Volume** | **None** | RunPod's "migrate to new host" flow has been known to attach a tiny 10 GB network volume — don't let it. We use container disk only |
| Expose | SSH (port 22, default) |  |

**Critical:** the field is named "Container Disk" — the persistent SSD. Do NOT confuse with "Network Volume" or "Volume Disk".

Click **Deploy On-Demand**. Wait ~30 sec for status `RUNNING`.

Copy the SSH command from **Connect → SSH over exposed TCP**. It looks like:
```text
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

#### Step 2 — SSH in

```bash
ssh root@<POD_IP> -p <POD_PORT> -i ~/.ssh/id_ed25519
```

Type `yes` on first connect. If `Permission denied`: `ssh-add ~/.ssh/id_ed25519`.

#### Step 3 — Clone + bootstrap (~10–15 min, all-in-one)

```bash
cd /workspace && \
git clone https://github.com/shiehn/sas-sample-generator.git && \
cd /workspace/sas-sample-generator && \
./scripts/setup.sh 2>&1 | tee /root/setup.log
```

Look for these "OK" markers near the end of setup.log:
```text
[setup]   rsync:          rsync version 3.x.x ...
[setup]   tmux:           tmux 3.x
[setup]   rubberband:     /usr/bin/rubberband
[setup]   ffmpeg:         ffmpeg version ...
[setup] cuda available: True
[setup] device:         NVIDIA RTX 4090
[setup] done.
```

If `cuda available: False` → you deployed onto a CPU template; terminate, redeploy with PyTorch GPU.

#### Step 4 — HF login + license acceptance

```bash
source /root/.venv/bin/activate
hf auth login
```

Paste your HF read token. Answer `n` to "Add token as git credential".

**First time on a new HF account:** in your browser, visit
[stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium)
and accept BOTH the SA3 community license **AND** the underlying Gemma
terms. Without both, the model download fails with `GatedRepoError`.
Token IS your account — accept while logged into the same HF account
your token belongs to.

Verify access (under 5 seconds):
```bash
hf download stabilityai/stable-audio-3-medium model_config.json --local-dir /tmp/sa3-test
ls /tmp/sa3-test/
```
If `model_config.json` is listed: cleared.

#### Step 5 — (Optional) Choose which categories to run

The default ships with **all 16** categories enabled. For a quick test
or a focused run:

```bash
# Edit scripts/pitched_categories.txt — comment out any category with `#`
nano scripts/pitched_categories.txt
```

After the test, restore with `git checkout scripts/pitched_categories.txt`.

#### Step 6 — Kick off the run inside tmux (~30 min generate + ~10 min gate)

```bash
tmux new -s pitched

# Inside tmux:
cd /workspace/sas-sample-generator
source /root/.venv/bin/activate
source /workspace/.bash_env

STAGES=generate,gate ./scripts/run_pitched.sh 2>&1 | tee /workspace/sas-sample-generator/outputs/run.log
```

**Detach with `Ctrl-b d`.** The run keeps going even if SSH drops.

**Reattach later** (from any new SSH session — possibly with a new IP/port if migrated):
```bash
tmux attach -t pitched
```

Monitor from outside tmux:
```bash
tail -f /workspace/sas-sample-generator/outputs/run.log
nvidia-smi
```

Per-prompt cost: ~1 sec generation × 5 variants × ~1500 prompts = ~2 hours of inference at SA3's 8 steps. The gate stage runs on CPU after generation — usually ~5–10 min for 7,500 variants.

#### Step 7 — Sanity-check the gate results

When `STAGES=generate,gate` finishes, before transferring:

```bash
# Per-category pass rates
for d in outputs/gated/*/; do
  cat=$(basename "$d")
  [[ "$cat" == "_failures" ]] && continue
  passed=$(ls "$d"*.wav 2>/dev/null | wc -l)
  failed=$(ls "$d/_failures"/*.json 2>/dev/null | wc -l)
  total=$((passed + failed))
  if [[ $total -gt 0 ]]; then
    rate=$((passed * 100 / total))
    printf "  %-18s passed=%3d  failed=%3d  pass-rate=%d%%\n" "$cat" "$passed" "$failed" "$rate"
  fi
done

echo "Total gated: $(find outputs/gated -name '*.wav' -not -path '*_failures*' | wc -l)"
du -sh outputs/gated
```

Expected (with the current thresholds, 2026-05-22): **80–100% pass rate per category**. If a category is below 50%, look in `outputs/gated/<cat>/_failures/<id>.json` to see why prompts are failing.

#### Step 8 — Pull data to your Mac via rsync

The pod has `rsync` installed by `setup.sh`. On your Mac:

```bash
mkdir -p ~/sas-pitched-out
rsync -avzP -e "ssh -p <POD_PORT> -i ~/.ssh/id_ed25519" \
  root@<POD_IP>:/workspace/sas-sample-generator/outputs/gated/ \
  ~/sas-pitched-out/gated/
```

For ~4 GB at typical RunPod / home upload speeds, expect 10–20 min.
`rsync` resumes on interruption — just re-run the same command if SSH drops.

Verify locally:
```bash
find ~/sas-pitched-out/gated -name '*.wav' -not -path '*_failures*' | wc -l   # should match step 7
du -sh ~/sas-pitched-out/gated
```

#### Step 9 — Run enrich locally (~30–60 min CPU-only)

```bash
cd ~/path/to/sas-sample-generator
git pull                                # pick up any threshold updates
pip install -r requirements.txt         # idempotent

export SAS_OUTPUTS_DIR=~/sas-pitched-out
STAGES=enrich ./scripts/run_pitched.sh
```

Each gated WAV → one instrument folder under `~/sas-pitched-out/enriched/<cat>/<id>/` containing:
- `sources/<midi>.wav` — root sample, smart-pitch-corrected + LUFS-normalized
- `zones/<midi>.flac` — pre-rendered chromatic zones (every 2–3 semitones)
- `manifest.json` — sampler-consumable schema
- `prompt.txt` — original positive prompt

#### Step 10 — ⚠️ TERMINATE THE POD

[runpod.io/console/pods](https://www.runpod.io/console/pods) → pod card → **Terminate** (NOT Stop). Compute billing stops immediately. Volume billing (if any auto-created Network Volume snuck in) stops only on Terminate.

Then [runpod.io/console/user/storage](https://www.runpod.io/console/user/storage) → **Network Volumes** → check for any `outside_*` orphan from a migration → Delete.

---

### Pod migration recovery (it happens)

RunPod sometimes moves your pod to a different physical host mid-run. Symptoms:
- SSH connection drops mid-session
- `Connection refused` when reconnecting on the same IP/port
- Pod shows "Stopped" briefly, then "Running" again at a new address

**The pod, the venv, the HF cache, and all `outputs/` data persist on the container disk** as long as the pod isn't terminated. You just need fresh connection info.

1. Open RunPod console → click your pod card → check the **Connect → SSH over exposed TCP** panel for the new IP and port (both can change).
2. Clear the old SSH host key on your Mac:
   ```bash
   ssh-keygen -R '[<NEW_IP>]:<NEW_PORT>'
   ```
3. SSH back in with the new details. Run `tmux attach -t pitched` — your run is still going.
4. If you were mid-rsync, just re-run the rsync command with the new `-p <NEW_PORT>` and `root@<NEW_IP>` — it picks up where it stopped.

This bit us twice this session (May 2026). Symptoms are unambiguous; recovery takes 30 seconds.

### Smart pitch correction (what enrich does)

SA3 doesn't reliably hit a target pitch from a text prompt — that's a known limitation of text-to-audio diffusion models. Enrich now compensates intelligently:

| If measured pitch is… | Enrich does… | Result |
|---|---|---|
| within `max_correction_semitones` of target (default 3) | shifts all the way to the original target | Sample is at exactly the prompted MIDI note; preserves prompt semantics |
| further away than that | snaps to the **nearest integer semitone** | Sample is at the closest "logical" MIDI note (always ≤50 cent shift, no audible artifacts) |

Either way: every output sample lands on an **exact MIDI semitone** with the smallest possible pitch shift. The zone rendering loop centers on that effective root, so the sampler always has a clean zone at the sample's actual pitch.

`max_correction_semitones` is per-category in `scripts/pitched_category_config.py`. Set to `0` to always snap to nearest semitone (never shift to target). Set to a large value (24+) to always shift to target.

### Gate stages explained

| Stage | What it checks | What rejection means |
|---|---|---|
| `prefilter` | Clipping, dead channels, all-silent buffers | Sample is broken at the file level |
| `onset` | Time from buffer start to first transient | `slow_onset` → SA3 added a fade-in / silence preamble (>300ms) |
| `sustain` | Longest plateau within 12 dB of peak RMS | `short_stab` → audio decays too fast or has no held region |
| `pitch` | CREPE periodicity + measured-vs-target | `no_voiced_frames` / `unconfident` → unpitched output; **`wrong_pitch` is OFF by default** (tolerance 9999) so enrich's snap-to-nearest-semitone can do its job |
| `polyphony` | BasicPitch note count after vibrato bypass | Disabled when TF/numpy ABI mismatches (common on RunPod) — gate prints one warning at start, then runs without it |

The gate scores winners by `confidence² × exp(-|cents|/50) × sus_quality`. With `wrong_pitch` disabled, the pitch term collapses to ~0 for far-off samples, so all variants of a prompt can tie at score=0.000 — the picker just grabs v00 by default in that case. Acceptable for now.

### Output layout

```
outputs/
├── raw/<category>/                                  ← SA3 generations (5 variants per prompt)
│   ├── <id>_v00.wav, <id>_v01.wav, ...
│   └── _metadata/<id>_v0N.json                      ← seed, model, generation_seconds
├── gated/<category>/                                ← gate winners only
│   ├── <id>.wav                                     ← chosen variant
│   ├── <id>.gate.json                               ← per-gate scores + measured pitch
│   └── _failures/<id>.json                          ← prompts where ALL variants rejected
└── enriched/<category>/<instrument-id>/             ← final library, sampler-consumable
    ├── sources/<midi>.wav                           ← effective-root-pitched, LUFS-normalized
    ├── zones/<midi>.flac                            ← pre-rendered chromatic zones
    ├── manifest.json                                ← v1 schema
    └── prompt.txt                                   ← original positive prompt
```

### Cost estimate (May 2026 prices)

For a full 16-category run with default 5 variants per prompt:

| Component | Time | Cost (RTX 4090 @ ~$0.34/hr) |
|---|---|---|
| Pod boot + setup.sh | ~10 min | $0.06 |
| HF model download (first call only) | ~3 min | $0.02 |
| Generate (~7,500 variants at ~1 sec each) | ~2 hr | $0.68 |
| Gate (CPU after generation) | ~10 min | $0.06 |
| Transfer + terminate | ~15 min | $0.09 |
| **Total on pod** | **~3 hr** | **~$0.91** |
| Local enrich | ~30–60 min on Mac | $0 |

Same on RTX A6000 (~$0.49/hr) → ~$1.30. On A100 (~$1.89/hr) → ~$5.00. The 4090 is the cheapest workable option.

### Authoring / iterating prompts

```
prompts/pitched/<category>.txt                # one prompt per line, # comments
scripts/pitched_categories.txt                # which categories to run (comment to skip)
scripts/pitched_category_config.py            # per-category target pitch, duration, sustain thresholds, etc.
```

Fast iteration on a single category:
```bash
# 1. Comment out 15 of 16 categories in scripts/pitched_categories.txt
# 2. Edit prompts/pitched/<cat>.txt
# 3. Re-run end-to-end
STAGES=generate,gate ./scripts/run_pitched.sh
# 4. Listen to outputs/gated/<cat>/*.wav, adjust prompts, re-run
```

`--skip-existing` in `batch_generate.py` means re-running won't regenerate samples you already have — only new prompt lines hit the GPU.

### Env knobs

- `STAGES=generate,gate,enrich` — comma-separated subset (default: all three)
- `STEPS=8` — diffusion steps (default 8; SA3 converges fast)
- `VARIANTS=5` — variants per prompt (default 5; vocals is bumped to 20 internally)
- `SAS_OUTPUTS_DIR=/some/path` — override outputs location (default `/workspace/outputs` on pod, `./outputs` local)

### What the plugin reads

The `sas-instrument-plugin` walks `outputs/enriched/<cat>/<id>/`, parses
each `manifest.json`, and uses the `zones[]` array to call
`host.setTrackInstrumentSampler` on the chosen track. Disjoint zones +
per-zone `root_midi` mean the engine pitch-shifts the nearest
pre-rendered zone for any played MIDI note, with the smart-corrected
sample as the unshifted root. Since enrich locks every sample to an
integer MIDI semitone, the sampler never has to deal with off-pitch
sources.

---

## When something breaks

| Symptom | Most likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` on ssh | private key not loaded into agent | `ssh-add ~/.ssh/id_ed25519` |
| `setup.sh` hangs at `Installing collected packages:` for >5 min | something redirected the venv onto `/workspace` (MooseFS); script defaults to `/root/.venv` for a reason | check `echo $VENV_DIR` — should be `/root/.venv`. If overridden, unset it and re-run |
| `cuda available: False` after `setup.sh` | picked a CPU template | terminate; re-deploy with PyTorch GPU template |
| `huggingface_hub.utils._errors.GatedRepoError` | didn't accept the SA3 license (or the Gemma terms it inherits) | visit the [model page](https://huggingface.co/stabilityai/stable-audio-3-medium), click "Agree" |
| `batch_generate.py` errors `CUDA out of memory` | duration too long for VRAM | lower `--default-duration` or `--num-waveforms-per-prompt 1` |
| All samples sound like loops | prompts not specific enough | add `one shot, no loop, no hi hats, no snare` to every prompt |
| Too much reverb | model adds ambience by default | add `dry, no reverb, no ambience` to prompts |
| Generated WAV doesn't sound like the target category (e.g. hats sound like snares) | the per-category `negative_prompt` may be excluding the target — bug in `scripts/category_config.py` | open `scripts/category_config.py`, audit the negative for that category; nothing in it should match the target sound |
| `run_all.sh` skips a category | corresponding `prompts/<cat>.txt` is missing or has only comments | check `prompts/` has the .txt file and contains non-comment lines |
| Samples land in `rejected/` with `reason=lufs_unmeasurable` | sample was so quiet that LUFS couldn't resolve even after loop-padding | usually a generation artifact — audition the rejected file. If a category has many of these, prompts are likely generating near-silent output; tighten them |
| One category sounds louder than the others in your sampler | likely peak-mode artifact, or LUFS target was overridden | re-measure the processed WAVs with `pyloudnorm` and check they cluster near `--target-lufs`. The pipeline log at `/workspace/run.log` (from step 6) records the as-run postprocess flags for each category. |
| SSH disconnects mid-run | network blip + foregrounded run | use `tmux new -s sas` BEFORE running, reattach with `tmux attach -t sas` |

---

## Cost recap

On an RTX A6000 at $0.49/hr. Both numbers below assume a fresh pod (so they
include the one-time bootstrap + model download). Stable Audio 3 needs only
~8 diffusion steps vs 120 for SAO 1.0, so inference time dropped ~15×
compared to the prior version of this repo.

| Run shape | Time | Cost |
|---|---|---|
| **Single category** (~100 samples) | ~7 min | ~$0.06 |
| **All 14 categories** (~1400 samples) | ~30 min | ~$0.25 |

The "all 14" cost is now dominated by the one-time bootstrap + model
download (~8 min) rather than inference (~1 sec × 1400 = ~25 min). If you
keep a pod alive between runs, subsequent iterations are pure inference and
the per-sample cost trends toward $0.0001.

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
- Why Stable Audio 3 vs alternatives (and what changed from SAO 1.0)
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
- [sas-stems-plugin](https://github.com/shiehn/sas-stems-plugin) — default AI audio-from-text + stem-splitting plugin
- [sas-loops-plugin](https://github.com/shiehn/sas-loops-plugin) — default audio loop / sample plugin
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
