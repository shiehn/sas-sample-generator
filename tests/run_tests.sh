#!/usr/bin/env bash
# CPU-only test suite for sas-sample-generator — runs on a Mac (M-series) or any
# machine with the venv deps. Does NOT touch Stable Audio / a GPU; it tests
# everything AROUND model generation: pitch detection, the drum + pitch gates,
# multi-source enrich, retry-to-target helpers, config/prompt/enable wiring,
# loudness-normalization targets, list_to_jsonl, the pitch report, and the
# deterministic pack builder.
#
#   ./tests/run_tests.sh
#
# Deps (already in the project .venv): numpy, soundfile, pyloudnorm, librosa,
# and the `rubberband` CLI (optional — the enrich test degrades gracefully).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer the project venv; fall back to python3 on PATH.
if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "ERROR: no python found (.venv/bin/python or python3)" >&2
  exit 1
fi
echo "Using: $("${PY}" --version 2>&1) (${PY})"
echo "rubberband: $(command -v rubberband >/dev/null 2>&1 && echo present || echo 'absent (enrich test degrades)')"
echo

MODULES=(
  tests/test_pitch_detection.py
  tests/test_drum_gate.py
  tests/test_enrich_multisource.py
  tests/test_pipeline_integration.py
)

PASS=0
FAIL=0
FAILED_MODULES=()
for m in "${MODULES[@]}"; do
  echo "================================================================"
  echo "▶ ${m}"
  echo "----------------------------------------------------------------"
  if "${PY}" "${m}"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    FAILED_MODULES+=("${m}")
  fi
  echo
done

echo "================================================================"
echo "Suite: ${PASS} module(s) passed, ${FAIL} failed."
if [[ ${FAIL} -gt 0 ]]; then
  printf '  FAILED: %s\n' "${FAILED_MODULES[@]}"
  exit 1
fi
echo "ALL GREEN ✅"
