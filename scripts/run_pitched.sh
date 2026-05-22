#!/usr/bin/env bash
# Run the full pitched-instrument pipeline:
#
#   1. Build prompts/pitched/<cat>.jsonl from prompts/pitched/<cat>.txt for every enabled category
#   2. Generate WAVs with a SINGLE batch_generate.py invocation (one pipeline load,
#      --num-waveforms-per-prompt 5 by default so the gate has variants to choose from)
#   3. Gate each category (quality + pitch + polyphony + sustain checks; picks the best variant)
#   4. Enrich each category (pitch-correct, LUFS-normalize, pre-render zones, write manifest)
#
# Enabled categories come from scripts/pitched_categories.txt — comment a line to skip.
#
# Outputs land under $SAS_OUTPUTS_DIR:
#   raw/<cat>/                  — SA3 generations (5 variants per prompt)
#   gated/<cat>/                — gate winners + sidecar gate.json + _failures/
#   instruments/<cat>/<id>/     — final library: source.wav, zones/<midi>.flac, manifest.json
#
# Tips:
#   STAGES=generate,gate ./scripts/run_pitched.sh   # GPU pod: skip enrich (CPU-bound)
#   STAGES=enrich         ./scripts/run_pitched.sh  # Local: enrich gated samples rsynced from pod
#   VARIANTS=3            ./scripts/run_pitched.sh  # cheaper test; gate has fewer to choose from
#   tmux new -s sas-pitched; ./scripts/run_pitched.sh   # survives SSH drops

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CATEGORIES_FILE="${REPO_ROOT}/scripts/pitched_categories.txt"
OUTPUTS_DIR="${SAS_OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
STEPS="${STEPS:-8}"                       # SA3 sweet spot
VARIANTS="${VARIANTS:-5}"                 # candidates per (prompt × target_pitch)
STAGES="${STAGES:-generate,gate,enrich}"  # comma-separated subset

if [[ ! -f "${CATEGORIES_FILE}" ]]; then
  echo "[run_pitched] ERROR: ${CATEGORIES_FILE} not found" >&2
  exit 1
fi

# Read non-empty, non-comment lines.
CATEGORIES=()
while IFS= read -r line; do
  line="${line%%#*}"
  line="${line## }"
  line="${line%% }"
  [[ -z "${line}" ]] && continue
  CATEGORIES+=("${line}")
done < "${CATEGORIES_FILE}"

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
  echo "[run_pitched] ERROR: no categories enabled in ${CATEGORIES_FILE}" >&2
  exit 1
fi

echo "[run_pitched] categories (${#CATEGORIES[@]}): ${CATEGORIES[*]}"
echo "[run_pitched] outputs dir: ${OUTPUTS_DIR}"
echo "[run_pitched] steps=${STEPS} variants=${VARIANTS} stages=${STAGES}"

want_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ---------- Stage 1: build JSONLs ----------
# Always run — JSONLs are cheap and the other stages depend on them. Done
# even when STAGES=enrich because enrich looks up prompt metadata.
JSONL_PATHS=()
for cat in "${CATEGORIES[@]}"; do
  txt="prompts/pitched/${cat}.txt"
  jsonl="prompts/pitched/${cat}.jsonl"
  if [[ ! -f "${txt}" ]]; then
    echo "[run_pitched] WARNING: ${txt} missing, skipping ${cat}" >&2
    continue
  fi
  echo "[run_pitched] building ${jsonl} <- ${txt}"
  python3 scripts/list_to_jsonl_pitched.py --in "${txt}" --out "${jsonl}"
  JSONL_PATHS+=("${jsonl}")
done

if [[ ${#JSONL_PATHS[@]} -eq 0 ]]; then
  echo "[run_pitched] ERROR: no JSONLs built — check prompts/pitched/<cat>.txt files exist" >&2
  exit 1
fi

# ---------- Stage 2: single batch_generate.py call for ALL JSONLs ----------
if want_stage generate; then
  mkdir -p "${OUTPUTS_DIR}"
  LOGFILE="${OUTPUTS_DIR}/batch_pitched.log"
  echo "[run_pitched] generating ${VARIANTS} variants per prompt for ${#JSONL_PATHS[@]} categories; log -> ${LOGFILE}"
  python scripts/batch_generate.py \
    --prompts "${JSONL_PATHS[@]}" \
    --out-root "${OUTPUTS_DIR}/raw" \
    --steps "${STEPS}" \
    --num-waveforms-per-prompt "${VARIANTS}" \
    --skip-existing 2>&1 | tee "${LOGFILE}"
else
  echo "[run_pitched] skipping generate (STAGES=${STAGES})"
fi

# ---------- Stage 3: gate each category ----------
if want_stage gate; then
  for cat in "${CATEGORIES[@]}"; do
    raw_dir="${OUTPUTS_DIR}/raw/${cat}"
    jsonl="prompts/pitched/${cat}.jsonl"
    if [[ ! -d "${raw_dir}" ]]; then
      echo "[run_pitched] skip gate ${cat} (no raw dir)"
      continue
    fi
    echo "[run_pitched] gating ${cat}"
    python scripts/gate_pitched.py \
      --category "${cat}" \
      --jsonl "${jsonl}" \
      --outputs-dir "${OUTPUTS_DIR}"
  done
else
  echo "[run_pitched] skipping gate (STAGES=${STAGES})"
fi

# ---------- Stage 4: enrich each category ----------
if want_stage enrich; then
  for cat in "${CATEGORIES[@]}"; do
    gated_dir="${OUTPUTS_DIR}/gated/${cat}"
    if [[ ! -d "${gated_dir}" ]]; then
      echo "[run_pitched] skip enrich ${cat} (no gated dir — did gate stage run?)"
      continue
    fi
    echo "[run_pitched] enriching ${cat}"
    python scripts/enrich_pitched.py --category "${cat}" --outputs-dir "${OUTPUTS_DIR}"
  done
else
  echo "[run_pitched] skipping enrich (STAGES=${STAGES})"
fi

echo ""
echo "[run_pitched] done."
echo "[run_pitched] raw:         ${OUTPUTS_DIR}/raw/<category>/"
echo "[run_pitched] gated:       ${OUTPUTS_DIR}/gated/<category>/"
echo "[run_pitched] instruments: ${OUTPUTS_DIR}/instruments/<category>/<instrument-id>/"
echo ""
echo "Cross-machine flow:"
echo "  pod:    STAGES=generate,gate ./scripts/run_pitched.sh"
echo "  rsync:  rsync -av <pod>:${OUTPUTS_DIR}/gated/ ./outputs/gated/"
echo "  local:  STAGES=enrich       ./scripts/run_pitched.sh"
