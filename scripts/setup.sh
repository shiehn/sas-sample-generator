#!/usr/bin/env bash
# First-time pod setup. Idempotent — safe to re-run on every pod boot.
#
# Layout (matters for speed):
#   /root/.venv               — Python venv. CONTAINER-LOCAL DISK (fast SSD).
#                               Lives ~30 seconds to install. Wiped on pod
#                               termination — that's fine, we rebuild fresh.
#   /workspace/.cache/...     — HuggingFace model cache. NETWORK FS (MooseFS).
#                               Slow for many-tiny-files but fine for the
#                               big sequential model downloads.
#   /workspace/outputs/...    — Generated audio. NETWORK FS.
#
# Why this split: RunPod's /workspace is a MooseFS network mount (verified
# via `df -T /workspace`). Many-tiny-file pip extraction over MooseFS takes
# ~15 minutes; the same install on /root takes ~30 seconds.
#
# Override any path with env vars: VENV_DIR=/foo ./scripts/setup.sh

set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/workspace}"
LOCAL_ROOT="${LOCAL_ROOT:-/root}"
PROJECT_DIR="${PROJECT_DIR:-${VOLUME_ROOT}/sas-sample-generator}"
VENV_DIR="${VENV_DIR:-${LOCAL_ROOT}/.venv}"
HF_CACHE="${HF_CACHE:-${VOLUME_ROOT}/.cache/huggingface}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

echo "[setup] volume:       ${VOLUME_ROOT}"
echo "[setup] project_dir:  ${PROJECT_DIR}"
echo "[setup] venv:         ${VENV_DIR}    (container-local, fast)"
echo "[setup] hf_cache:     ${HF_CACHE}"

if [[ ! -d "${VOLUME_ROOT}" ]]; then
  echo "[setup] ERROR: ${VOLUME_ROOT} is not mounted." >&2
  echo "[setup]        Attach a 100 GB Volume Disk at /workspace when deploying the pod." >&2
  exit 1
fi

if [[ ! -d "${LOCAL_ROOT}" ]] || [[ ! -w "${LOCAL_ROOT}" ]]; then
  echo "[setup] ERROR: ${LOCAL_ROOT} is not writable. Venv would be installed somewhere slow." >&2
  echo "[setup]        Override with: VENV_DIR=/some/writable/path ./scripts/setup.sh" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}" "${HF_CACHE}" "${VOLUME_ROOT}/outputs"

# Persist HF cache + outputs paths across shells.
PROFILE="${VOLUME_ROOT}/.bash_env"
cat > "${PROFILE}" <<EOF
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE}/hub"
export SAS_OUTPUTS_DIR="${VOLUME_ROOT}/outputs"
export SAS_VENV_DIR="${VENV_DIR}"
EOF
# shellcheck disable=SC1090
source "${PROFILE}"

# Make sure interactive shells pick this up next time.
if ! grep -q "${PROFILE}" "${HOME}/.bashrc" 2>/dev/null; then
  echo "source ${PROFILE}" >> "${HOME}/.bashrc"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] creating venv at ${VENV_DIR}"
  python -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

# Install torch from the CUDA wheel index; skip if already present and matches.
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[setup] installing torch + torchaudio from ${TORCH_CUDA_INDEX}"
  pip install torch torchaudio --index-url "${TORCH_CUDA_INDEX}"
fi

REQ_FILE="${PROJECT_DIR}/requirements.txt"
if [[ -f "${REQ_FILE}" ]]; then
  echo "[setup] installing ${REQ_FILE}"
  pip install -r "${REQ_FILE}"
fi

python - <<'PY'
import torch
print(f"[setup] cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] device:         {torch.cuda.get_device_name(0)}")
PY

echo ""
echo "[setup] done."
echo "[setup] next: source ${VENV_DIR}/bin/activate"
echo "[setup]       hf auth login"
