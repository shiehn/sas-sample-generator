# sas-sample-generator

Generate batches of one-shot audio samples (kicks, snares, hats, etc.) with
[Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0)
on a rented [RunPod](https://www.runpod.io) GPU.

**Designed for occasional use.** This README is the recipe — read top to bottom,
copy-paste each command block, finish in ~30 minutes for ~$0.30 of GPU time.

Assumes an **Apple Silicon Mac** as the control machine.

For the rationale (why these settings, prompt-design tips, deep cost math),
see [`stable_audio_open_batch_oneshot_guide.md`](stable_audio_open_batch_oneshot_guide.md).

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

### 3. Clone + bootstrap (SLOW: ~10–15 min on first boot)

On the pod:
```bash
cd /workspace && \
git clone https://github.com/shiehn/sas-sample-generator.git && \
cd sas-sample-generator && \
./scripts/setup.sh 2>&1 | tee /workspace/setup.log
```

⚠️ **Looks frozen at `Installing collected packages:` for ~10 min — this is normal.**
`/workspace` is a network filesystem (MooseFS) and pip is extracting ~4 GB of
CUDA libraries one tiny file at a time. Don't Ctrl-C.

You're done when you see:
```text
[setup] cuda available: True
[setup] device:         NVIDIA RTX A6000
[setup] done. ...
```

### 4. Hugging Face login

```bash
source /workspace/.venv/bin/activate
huggingface-cli login
```

Paste your HF token (One-Time Setup A). Answer `n` to "Add token as git
credential".

### 5. (Optional) Write your own prompts list

The repo ships with `prompts/kicks.txt` (28 kick-drum prompts across 10
categories). **Use it as-is to test, or skip ahead to step 6.**

To make your own: edit `prompts/<category>.txt` — one description per line,
`#` for comments:

```text
# 808-style
deep 808 kick one shot, long sub bass decay, dry, no melody, no loop
warm 808 kick one shot, saturated low end, medium decay, dry

# Lo-fi
warm lo-fi kick one shot, soft transient, dusty sampler texture, dry
```

Use `nano prompts/kicks.txt` on the pod, or edit locally and `scp` it over,
or edit + `git push` and `git pull` on the pod.

### 6. Convert the text list to JSONL

```bash
python3 scripts/list_to_jsonl.py \
  --in prompts/kicks.txt \
  --out prompts/kicks.jsonl \
  --prefix kick
```

Use `--prefix kick` for kick IDs (`kick_0001`, `kick_0002`, …). Switch to
`--prefix snare` etc. for other categories.

### 7. Generate the audio (~5–10 min for 30 prompts)

```bash
python scripts/batch_generate.py \
  --prompts prompts/kicks.jsonl \
  --out /workspace/outputs/raw \
  --steps 120 \
  --skip-existing 2>&1 | tee /workspace/batch.log
```

First call downloads Stable Audio Open (~3–5 GB, ~3 min). Then a tqdm bar at
~10 sec/sample. The `--skip-existing` flag lets you re-run safely if anything
dies mid-batch.

### 8. Post-process

```bash
python scripts/postprocess_oneshots.py \
  --in-dir /workspace/outputs/raw \
  --out-dir /workspace/outputs/processed \
  --rejected-dir /workspace/outputs/rejected \
  --manifest /workspace/outputs/manifests/run_manifest.csv \
  --mono \
  --max-seconds 2.5
```

Trims silence, normalizes peaks, downmixes to mono, writes a manifest CSV.
Last line tells you `Processed: N`, `Rejected: M`.

### 9. Zip and download

On the pod:
```bash
cd /workspace
zip -r run.zip outputs/processed outputs/manifests outputs/raw/_metadata
ls -lh run.zip
```

In a **second** Mac terminal (don't close the SSH session yet — you still
need it for step 10):

```bash
cd ~/Downloads
scp -P <POD_PORT> root@<POD_IP>:/workspace/run.zip .
unzip run.zip
open outputs/processed                 # Finder + QuickLook to audition
```

`<POD_PORT>` and `<POD_IP>` are the same ones from your step-1 SSH command.

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
│   ├── kicks.txt                               ← starter kick-drum prompts (28)
│   ├── kicks.jsonl                             ← generated by list_to_jsonl.py
│   └── kicks_smoke_test.jsonl                  ← 3 prompts for a free wire-check
├── scripts/
│   ├── setup.sh                                ← step 3
│   ├── list_to_jsonl.py                        ← step 6
│   ├── batch_generate.py                       ← step 7
│   ├── postprocess_oneshots.py                 ← step 8
│   ├── benchmark.py                            ← optional: per-sample cost math
│   └── sync.sh                                 ← optional: rclone to B2 / R2
└── outputs/                                    ← gitignored; generated WAVs land here
```

---

## When something breaks

| Symptom | Most likely cause | Fix |
|---|---|---|
| `Permission denied (publickey)` on ssh | private key not loaded into agent | `ssh-add ~/.ssh/id_ed25519` |
| `setup.sh` hangs at `Installing collected packages:` | MooseFS network FS is slow; not stuck | wait 10–15 min |
| `cuda available: False` after `setup.sh` | picked a CPU template | terminate; re-deploy with PyTorch GPU template |
| `huggingface_hub.utils._errors.GatedRepoError` | didn't accept the SAO license | visit the [model page](https://huggingface.co/stabilityai/stable-audio-open-1.0), click "Agree" |
| `batch_generate.py` errors `CUDA out of memory` | duration too long for VRAM | lower `--default-duration` or `--num-waveforms-per-prompt 1` |
| All samples sound like loops | prompts not specific enough | add `one shot, no loop, no hi hats, no snare` to every prompt |
| Too much reverb | model adds ambience by default | add `dry, no reverb, no ambience` to prompts |

---

## Cost recap

A single run, on an RTX A6000 at $0.49/hr:

| Phase | Time | Cost |
|---|---|---|
| Pod boot + `setup.sh` | ~15 min | ~$0.12 |
| HF model download (first call only) | ~3 min | ~$0.02 |
| Generate ~30 samples | ~5 min | ~$0.04 |
| Post-process + zip + scp | ~2 min | ~$0.02 |
| **One-batch total** | **~25 min** | **~$0.20** |

If you keep the pod alive between runs in the same session (e.g., iterating on
prompts), each subsequent generation is just step 7 again — no setup cost.

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
