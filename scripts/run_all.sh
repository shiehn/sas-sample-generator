#!/usr/bin/env bash
# Run the full multi-category sample-generation pipeline:
#   1. Build prompts/<cat>.jsonl from prompts/<cat>.txt for every enabled category
#      (each row carries a per-category "variants" count from drum_gate_config.py)
#   2. Generate N variants/prompt with a SINGLE batch_generate.py invocation (one model load)
#   2b. Quality-gate + best-of-N select (gate_drums.py: raw -> gated_drums) when GATE=1
#   3. Post-process the winners (trim / LUFS / mono / tags) -> processed/
#
# Enabled categories come from scripts/categories.txt — comment a line to skip.
#
# Outputs land under $SAS_OUTPUTS_DIR (set by scripts/setup.sh; defaults to
# /workspace/outputs on a RunPod pod, or ./outputs locally).
#
# Tips:
#   STEPS=4 ./scripts/run_all.sh           # even cheaper iteration (SA3 needs few steps)
#   ONLY="kick clap" LIMIT=10 ./scripts/run_all.sh   # small test slice
#   GATE=0 ./scripts/run_all.sh            # skip the quality gate (legacy: keep all from raw)
#   STAGES=gate,postprocess ./scripts/run_all.sh  # re-gate EXISTING raw (no GPU) + postprocess
#   BATCH_SIZE=32 ./scripts/run_all.sh     # bigger batches on an 80GB GPU
#   tmux new -s sas; ./scripts/run_all.sh  # survives SSH drops

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
BATCH_SIZE="${BATCH_SIZE:-16}"   # generations per model call (32-64 on a big GPU)
ONLY="${ONLY:-}"                 # space/comma list to override enabled categories (test slice)
LIMIT="${LIMIT:-}"               # cap prompts/category (e.g. ONLY=clap LIMIT=10)
GATE="${GATE:-1}"                # 1 = quality-gate + best-of-N (raw->gated_drums->processed);
                                 # 0 = legacy (postprocess straight from raw, keep all)
MAX_RETRIES="${MAX_RETRIES:-2}"  # re-roll rounds for failed prompts before topping up
TARGET="${TARGET:-150}"          # per-category MINIMUM surviving samples (0 = none)
[[ -n "${LIMIT}" ]] && TARGET=0  # small test slice must not chase the full minimum
STAGES="${STAGES:-generate,gate,postprocess}"  # comma-subset of stages to run.
                                 # STAGES=gate,postprocess re-gates EXISTING raw with
                                 # NO regeneration (reuse everything already generated).
want_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

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

# ONLY overrides the enabled list — e.g. `ONLY="kick clap" LIMIT=10 ./scripts/run_all.sh`.
if [[ -n "${ONLY}" ]]; then
  IFS=', ' read -r -a CATEGORIES <<< "${ONLY}"
fi

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
  echo "[run_all] ERROR: no categories enabled in ${CATEGORIES_FILE}" >&2
  exit 1
fi

echo "[run_all] categories (${#CATEGORIES[@]}): ${CATEGORIES[*]}"
echo "[run_all] outputs dir: ${OUTPUTS_DIR}"
echo "[run_all] steps:       ${STEPS}${LIMIT:+ limit=${LIMIT}}"

# ---------- Step 1: build JSONLs from .txt files ----------
JSONL_PATHS=()
for cat in "${CATEGORIES[@]}"; do
  txt="prompts/${cat}.txt"
  jsonl="prompts/${cat}.jsonl"
  if [[ ! -f "${txt}" ]]; then
    echo "[run_all] WARNING: ${txt} missing, skipping ${cat}" >&2
    continue
  fi
  echo "[run_all] building ${jsonl} <- ${txt}${LIMIT:+ (limit ${LIMIT})}"
  python3 scripts/list_to_jsonl.py --in "${txt}" --out "${jsonl}" ${LIMIT:+--limit "${LIMIT}"}
  JSONL_PATHS+=("${jsonl}")
done

if [[ ${#JSONL_PATHS[@]} -eq 0 ]]; then
  echo "[run_all] ERROR: no JSONLs built — check that prompts/<cat>.txt files exist" >&2
  exit 1
fi

# ---------- Step 2 (+2b): generate, then quality-gate + best-of-N selection ----------
mkdir -p "${OUTPUTS_DIR}"
LOGFILE="${OUTPUTS_DIR}/batch.log"
if want_stage generate && want_stage gate && [[ "${GATE}" == "1" ]]; then
  # run_retry drives generate + gate_drums together, regenerating any prompt
  # whose candidates all fail until the gate yields a sample (up to MAX_RETRIES).
  # batch_generate runs with --skip-existing, so raw already on disk is REUSED
  # (never regenerated) — only missing / short-of-target prompts hit the GPU.
  echo "[run_all] generate + gate (best-of-N, target=${TARGET}/cat, max_retries=${MAX_RETRIES}); log -> ${LOGFILE}"
  python3 scripts/run_retry.py \
    --pipeline drums \
    --categories "${CATEGORIES[@]}" \
    --outputs-dir "${OUTPUTS_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --target "${TARGET}" \
    --max-retries "${MAX_RETRIES}" 2>&1 | tee "${LOGFILE}"
elif want_stage gate && ! want_stage generate; then
  # Re-gate EXISTING raw with NO generation: reuse everything already generated
  # (e.g. after a gate-logic fix). gate_drums re-evaluates every raw variant and
  # re-selects winners into gated_drums/. Pure CPU — the GPU is never touched.
  echo "[run_all] re-gating existing raw, no generation (STAGES=${STAGES}); log -> ${LOGFILE}"
  : > "${LOGFILE}"
  for cat in "${CATEGORIES[@]}"; do
    if [[ ! -d "${OUTPUTS_DIR}/raw/${cat}" ]]; then
      echo "[run_all] skip gate ${cat} (no ${OUTPUTS_DIR}/raw/${cat})" | tee -a "${LOGFILE}"
      continue
    fi
    python3 scripts/gate_drums.py --category "${cat}" --outputs-dir "${OUTPUTS_DIR}" 2>&1 | tee -a "${LOGFILE}"
  done
elif want_stage generate; then
  # Generate only, no gate (GATE=0 keep-all, or STAGES without 'gate').
  echo "[run_all] generating only, no gate (STAGES=${STAGES}, GATE=${GATE}); log -> ${LOGFILE}"
  python scripts/batch_generate.py \
    --prompts "${JSONL_PATHS[@]}" \
    --out-root "${OUTPUTS_DIR}/raw" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --skip-existing 2>&1 | tee "${LOGFILE}"
else
  echo "[run_all] skipping generate + gate (STAGES=${STAGES})"
fi

# ---------- Step 3: post-process each category ----------
# With GATE=1 we post-process the gated winners; otherwise straight from raw.
if want_stage postprocess; then
  for cat in "${CATEGORIES[@]}"; do
    if [[ "${GATE}" == "1" ]]; then
      src_dir="${OUTPUTS_DIR}/gated_drums/${cat}"
    else
      src_dir="${OUTPUTS_DIR}/raw/${cat}"
    fi
    if [[ ! -d "${src_dir}" ]]; then
      echo "[run_all] skip postprocess ${cat} (no ${src_dir})"
      continue
    fi
    echo "[run_all] post-processing ${cat}"
    python scripts/postprocess_oneshots.py --category "${cat}" --in-dir "${src_dir}" --mono
  done
else
  echo "[run_all] skipping postprocess (STAGES=${STAGES})"
fi

echo ""
echo "[run_all] done."
echo "[run_all] WAVs under: ${OUTPUTS_DIR}/processed/<category>/"
echo "[run_all] manifests:  ${OUTPUTS_DIR}/manifests/<category>.csv"
