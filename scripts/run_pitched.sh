#!/usr/bin/env bash
# Run the full pitched-instrument pipeline:
#
#   1. Build prompts/pitched/<cat>.jsonl from prompts/pitched/<cat>.txt for every enabled category
#   2. Generate WAVs with a SINGLE batch_generate.py invocation (one model load,
#      BATCHED; per-category variant counts come from pitched_category_config.py
#      via each JSONL row's "variants" field, so the gate has candidates to pick)
#   3. Gate each category (quality + pitch + polyphony + sustain checks; picks the best variant)
#   3b. Report pitch accuracy (outputs/_reports/pitch_summary.{json,md}) — measured-vs-target
#   4. Enrich each category (pitch-correct, LUFS-normalize, pre-render zones, write manifest)
#
# Enabled categories come from scripts/pitched_categories.txt — comment a line to skip.
#
# Outputs land under $SAS_OUTPUTS_DIR:
#   raw/<cat>/                  — SA3 generations (per-category variants per prompt)
#   gated/<cat>/                — gate winners + sidecar gate.json + _failures/
#   instruments/<cat>/<id>/     — final library: source.wav, zones/<midi>.flac, manifest.json
#
# Tips:
#   STAGES=generate,gate,report ./scripts/run_pitched.sh   # GPU pod: skip enrich (CPU-bound)
#   STAGES=enrich               ./scripts/run_pitched.sh   # Local: enrich gated samples rsynced from pod
#   STAGES=report               ./scripts/run_pitched.sh   # Re-emit the pitch report from existing gate.json
#   BATCH_SIZE=32        ./scripts/run_pitched.sh  # bigger batches on a 80GB GPU
#   ONLY=pianos LIMIT=5  ./scripts/run_pitched.sh  # SMALL test slice: 5 piano prompts
#                                                  # (multi-source -> a few playable pianos to
#                                                  #  check pitch/temperament across the keyboard)
#   ONLY="basses synths" LIMIT=20 STAGES=generate,gate,report ./scripts/run_pitched.sh  # pitch A/B slice
#   MAX_RETRIES=3        ./scripts/run_pitched.sh  # more retry rounds to hit the count
#   MAX_RETRIES=0        ./scripts/run_pitched.sh  # disable retry (one pass, drop failures)
#   tmux new -s sas-pitched; ./scripts/run_pitched.sh   # survives SSH drops

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Source /workspace/.bash_env if SAS_OUTPUTS_DIR isn't already set in the
# environment — same fix as run_all.sh. Without this, run_pitched.sh from
# a fresh tmux pane / new SSH session that didn't inherit interactive-shell
# env can fall back to ${REPO_ROOT}/outputs while a sibling run in a
# different shell uses /workspace/outputs. Cost us a couple of hours.
if [[ -z "${SAS_OUTPUTS_DIR:-}" && -f /workspace/.bash_env ]]; then
  # shellcheck disable=SC1091
  source /workspace/.bash_env
fi

CATEGORIES_FILE="${REPO_ROOT}/scripts/pitched_categories.txt"
OUTPUTS_DIR="${SAS_OUTPUTS_DIR:-${REPO_ROOT}/outputs}"
STEPS="${STEPS:-8}"                       # SA3 sweet spot
BATCH_SIZE="${BATCH_SIZE:-16}"            # generations per model call (32-64 on 80GB GPU)
STAGES="${STAGES:-generate,gate,enrich,report}"  # comma-separated subset
ONLY="${ONLY:-}"                          # space/comma list to override the enabled categories
LIMIT="${LIMIT:-}"                        # cap prompts/category (small test slice, e.g. LIMIT=5)
MAX_RETRIES="${MAX_RETRIES:-2}"           # retry rounds for prompts whose variants all fail the gate
INIT_ANCHOR="${INIT_ANCHOR:-}"            # set to 1 to enable init_audio pitch anchoring (experiment)

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

# ONLY overrides the enabled list — for a small test slice on one or a few
# categories, e.g. `ONLY=pianos LIMIT=5 ./scripts/run_pitched.sh`.
if [[ -n "${ONLY}" ]]; then
  IFS=', ' read -r -a CATEGORIES <<< "${ONLY}"
fi

if [[ ${#CATEGORIES[@]} -eq 0 ]]; then
  echo "[run_pitched] ERROR: no categories enabled in ${CATEGORIES_FILE}" >&2
  exit 1
fi

echo "[run_pitched] categories (${#CATEGORIES[@]}): ${CATEGORIES[*]}"
echo "[run_pitched] outputs dir: ${OUTPUTS_DIR}"
echo "[run_pitched] steps=${STEPS} batch_size=${BATCH_SIZE} stages=${STAGES}${LIMIT:+ limit=${LIMIT}} (per-category variants from config)"

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
  echo "[run_pitched] building ${jsonl} <- ${txt}${LIMIT:+ (limit ${LIMIT})}"
  python3 scripts/list_to_jsonl_pitched.py --in "${txt}" --out "${jsonl}" ${LIMIT:+--limit "${LIMIT}"}
  JSONL_PATHS+=("${jsonl}")
done

if [[ ${#JSONL_PATHS[@]} -eq 0 ]]; then
  echo "[run_pitched] ERROR: no JSONLs built — check prompts/pitched/<cat>.txt files exist" >&2
  exit 1
fi

# ---------- Stage 2+3: generate + gate (with retry-to-target) ----------
# When BOTH are wanted (the normal pod flow) run_retry.py drives them together,
# regenerating any prompt whose candidates all fail until the gate yields a
# sample (up to MAX_RETRIES rounds). generate-only / gate-only stay as simple
# fallbacks (no retry — there's nothing to react to).
mkdir -p "${OUTPUTS_DIR}"
LOGFILE="${OUTPUTS_DIR}/batch_pitched.log"
if want_stage generate && want_stage gate; then
  echo "[run_pitched] generate+gate with retry-to-target (max_retries=${MAX_RETRIES}); log -> ${LOGFILE}"
  python3 scripts/run_retry.py \
    --pipeline pitched \
    --categories "${CATEGORIES[@]}" \
    --outputs-dir "${OUTPUTS_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --max-retries "${MAX_RETRIES}" \
    ${INIT_ANCHOR:+--init-audio-anchor} 2>&1 | tee "${LOGFILE}"
elif want_stage generate; then
  echo "[run_pitched] generate only (no gate -> no retry)"
  python scripts/batch_generate.py --prompts "${JSONL_PATHS[@]}" \
    --out-root "${OUTPUTS_DIR}/raw" --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
    ${INIT_ANCHOR:+--init-audio-anchor} --skip-existing 2>&1 | tee "${LOGFILE}"
elif want_stage gate; then
  echo "[run_pitched] gate only (no retry)"
  for cat in "${CATEGORIES[@]}"; do
    [[ -d "${OUTPUTS_DIR}/raw/${cat}" ]] || { echo "[run_pitched] skip gate ${cat} (no raw dir)"; continue; }
    python scripts/gate_pitched.py --category "${cat}" \
      --jsonl "prompts/pitched/${cat}.jsonl" --outputs-dir "${OUTPUTS_DIR}"
  done
else
  echo "[run_pitched] skipping generate+gate (STAGES=${STAGES})"
fi

# ---------- Stage 3b: pitch-accuracy report (cheap, CPU-only) ----------
# Reads the gate sidecars and emits ${OUTPUTS_DIR}/_reports/pitch_summary.{json,md}.
# Run on the pod right after gate so the measured-vs-target accuracy is visible
# before spending the rest of the campaign / before transferring for enrich.
if want_stage report; then
  echo "[run_pitched] building pitch-accuracy report for ${#CATEGORIES[@]} categories"
  python3 scripts/pitch_report.py \
    --outputs-dir "${OUTPUTS_DIR}" \
    --categories "${CATEGORIES[@]}" || echo "[run_pitched] WARNING: pitch_report failed (non-fatal)" >&2
else
  echo "[run_pitched] skipping report (STAGES=${STAGES})"
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
