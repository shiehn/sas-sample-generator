#!/usr/bin/env bash
# One-command full v3 campaign on a GPU pod: drums + pitched instruments,
# ALL stages on the pod (pitched enrich INCLUDED), so the only thing left to do
# afterward is rsync the final library (processed/ + instruments/) to your Mac.
#
# Why everything on the pod: enrich is CPU-only (rubberband-cli, installed by
# setup.sh), so it runs fine here — the trade-off is the GPU idles for ~30-45min
# during enrich. The payoff is a single final download and a Mac that needs
# nothing but rsync.
#
# Disk: kept raw/ candidates dominate (~110-155GB); pod-side enrich adds the
# ~20-24GB instruments/ library on top. Budget ~300GB container disk. This
# script refuses to start if free space looks too small (override MIN_FREE_GB).
#
# Usage (inside tmux, after setup.sh + `hf auth login`):
#   ./scripts/full_run.sh
#
# Honors the same env knobs as run_all.sh / run_pitched.sh, with pod defaults:
#   BATCH_SIZE        (default 32; 48-64 ok on an 80GB GPU)
#   SAS_MULTI_SOURCE  (default 1)
#   STEPS TARGET MAX_RETRIES ONLY LIMIT   (passed straight through)
#   SAS_OUTPUTS_DIR   (default from /workspace/.bash_env, else ./outputs)
#   MIN_FREE_GB       (preflight floor, default 250)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Resolve SAS_OUTPUTS_DIR the same way the run scripts do.
if [[ -z "${SAS_OUTPUTS_DIR:-}" && -f /workspace/.bash_env ]]; then
  # shellcheck disable=SC1091
  source /workspace/.bash_env
fi
OUTPUTS_DIR="${SAS_OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SAS_MULTI_SOURCE="${SAS_MULTI_SOURCE:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-250}"

mkdir -p "${OUTPUTS_DIR}"

# ---------- Preflight: free disk on the outputs filesystem ----------
avail_kb="$(df -Pk "${OUTPUTS_DIR}" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "${avail_kb}" ]]; then
  echo "[full_run] WARN: couldn't read free disk for ${OUTPUTS_DIR}; skipping preflight" >&2
else
  avail_gb=$(( avail_kb / 1024 / 1024 ))
  echo "[full_run] outputs dir:     ${OUTPUTS_DIR}"
  echo "[full_run] free disk there: ${avail_gb} GB  (need >= ${MIN_FREE_GB} GB for a full kept-raw run)"
  if (( avail_gb < MIN_FREE_GB )); then
    echo "[full_run] ERROR: only ${avail_gb} GB free; a full run needs ~${MIN_FREE_GB} GB." >&2
    echo "[full_run]   fixes: resize the container disk, OR run a subset (ONLY=...) and" >&2
    echo "[full_run]   delete outputs/raw between waves, OR override with MIN_FREE_GB=<n>." >&2
    exit 1
  fi
fi

echo "[full_run] BATCH_SIZE=${BATCH_SIZE} SAS_MULTI_SOURCE=${SAS_MULTI_SOURCE}"
echo ""
echo "[full_run] === STAGE 1/2: drums (generate -> gate -> postprocess) ==="
./scripts/run_all.sh 2>&1 | tee "${OUTPUTS_DIR}/drums.log"

echo ""
echo "[full_run] === STAGE 2/2: pitched (generate -> gate -> report -> enrich) ==="
./scripts/run_pitched.sh 2>&1 | tee "${OUTPUTS_DIR}/pitched.log"

# ---------- Summary ----------
drum_wavs="$(find "${OUTPUTS_DIR}/processed" -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')"
instruments="$(find "${OUTPUTS_DIR}/instruments" -name 'manifest.json' 2>/dev/null | wc -l | tr -d ' ')"
echo ""
echo "[full_run] DONE. Final library:"
echo "  drums:       ${OUTPUTS_DIR}/processed/    (${drum_wavs} wavs)"
echo "  instruments: ${OUTPUTS_DIR}/instruments/  (${instruments} instruments)"
du -sh "${OUTPUTS_DIR}/processed" "${OUTPUTS_DIR}/instruments" 2>/dev/null || true
echo ""
echo "[full_run] Next: rsync processed/ + instruments/ to your Mac, then TERMINATE the pod:"
echo "  rsync -avzP -e 'ssh -p <PORT> -i ~/.ssh/id_ed25519' root@<IP>:${OUTPUTS_DIR}/processed/   ~/sas-out/processed/"
echo "  rsync -avzP -e 'ssh -p <PORT> -i ~/.ssh/id_ed25519' root@<IP>:${OUTPUTS_DIR}/instruments/ ~/sas-out/instruments/"
