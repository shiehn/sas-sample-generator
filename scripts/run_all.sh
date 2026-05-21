#!/usr/bin/env bash
# Run the full multi-category sample-generation pipeline:
#   1. Build prompts/<cat>.jsonl from prompts/<cat>.txt for every enabled category
#   2. Generate WAVs with a SINGLE batch_generate.py invocation (one pipeline load)
#   3. Post-process each category in turn
#
# Enabled categories come from scripts/categories.txt — comment a line to skip.
#
# Outputs land under $SAS_OUTPUTS_DIR (set by scripts/setup.sh; defaults to
# /workspace/outputs on a RunPod pod, or ./outputs locally).
#
# Tips:
#   STEPS=4 ./scripts/run_all.sh       # even cheaper iteration (SA3 needs few steps)
#   tmux new -s sas; ./scripts/run_all.sh   # survives SSH drops

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# setup.sh writes /workspace/.bash_env with SAS_OUTPUTS_DIR=/workspace/outputs
# and adds a source line to ~/.bashrc. That only fires for interactive shells
# — tmux panes / scripted SSH sessions can launch without it, which silently
# falls back to ${REPO_ROOT}/outputs and lands the WAVs in the wrong place
# (lost an entire 14-category run to this bug once). Source explicitly here.
if [[ -z "${SAS_OUTPUTS_DIR:-}" && -f /workspace/.bash_env ]]; then
  # shellcheck disable=SC1091
  source /workspace/.bash_env
fi

CATEGORIES_FILE="${REPO_ROOT}/scripts/categories.txt"
OUTPUTS_DIR="${SAS_OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
STEPS="${STEPS:-8}"

if [[ ! -f "${CATEGORIES_FILE}" ]]; then
  echo "[run_all] ERROR: ${CATEGORIES_FILE} not found" >&2
  exit 1
fi

# Read non-empty, non-comment lines.
CATEGORIES=()
while IFS= read -r line; do
  line="${line%%#*}"        # strip inline comments
  line="${line## }"
  line="${line%% }"
  [[ -z "${line}" ]] && continue
  CATEGORIES+=("${line}")
done < "${CATEGORIES_FILE}"

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
  echo "[run_all] ERROR: no categories enabled in ${CATEGORIES_FILE}" >&2
  exit 1
fi

echo "[run_all] categories (${#CATEGORIES[@]}): ${CATEGORIES[*]}"
echo "[run_all] outputs dir: ${OUTPUTS_DIR}"
echo "[run_all] steps:       ${STEPS}"

# ---------- Step 1: build JSONLs from .txt files ----------
JSONL_PATHS=()
for cat in "${CATEGORIES[@]}"; do
  txt="prompts/${cat}.txt"
  jsonl="prompts/${cat}.jsonl"
  if [[ ! -f "${txt}" ]]; then
    echo "[run_all] WARNING: ${txt} missing, skipping ${cat}" >&2
    continue
  fi
  echo "[run_all] building ${jsonl} <- ${txt}"
  python3 scripts/list_to_jsonl.py --in "${txt}" --out "${jsonl}"
  JSONL_PATHS+=("${jsonl}")
done

if [[ ${#JSONL_PATHS[@]} -eq 0 ]]; then
  echo "[run_all] ERROR: no JSONLs built — check that prompts/<cat>.txt files exist" >&2
  exit 1
fi

# ---------- Step 2: single batch_generate.py call for ALL JSONLs ----------
mkdir -p "${OUTPUTS_DIR}"
LOGFILE="${OUTPUTS_DIR}/batch.log"
echo "[run_all] generating samples for ${#JSONL_PATHS[@]} categories; log -> ${LOGFILE}"
python scripts/batch_generate.py \
  --prompts "${JSONL_PATHS[@]}" \
  --out-root "${OUTPUTS_DIR}/raw" \
  --steps "${STEPS}" \
  --skip-existing 2>&1 | tee "${LOGFILE}"

# ---------- Step 3: post-process each category ----------
for cat in "${CATEGORIES[@]}"; do
  raw_dir="${OUTPUTS_DIR}/raw/${cat}"
  if [[ ! -d "${raw_dir}" ]]; then
    echo "[run_all] skip postprocess ${cat} (no raw dir)"
    continue
  fi
  echo "[run_all] post-processing ${cat}"
  python scripts/postprocess_oneshots.py --category "${cat}" --mono
done

echo ""
echo "[run_all] done."
echo "[run_all] WAVs under: ${OUTPUTS_DIR}/processed/<category>/"
echo "[run_all] manifests:  ${OUTPUTS_DIR}/manifests/<category>.csv"
