#!/usr/bin/env bash
# Build the two ready-to-consume sample libraries (drums + instruments) as
# DIRECTORIES with the _pack-version.json marker — droppable straight into the
# app's <userData>/sample-packs/{drums,instruments}/ with no zip/download step.
#
# Run AFTER gating + enriching, i.e. once you have:
#   $SAS_OUTPUTS_DIR/processed/    (drums, from run_all.sh)
#   $SAS_OUTPUTS_DIR/instruments/  (instruments, from run_pitched.sh enrich)
#
# Usage:
#   DRUM_VERSION=3 INSTRUMENT_VERSION=3 ./scripts/build_libraries.sh
#   FMT=both DRUM_VERSION=3 INSTRUMENT_VERSION=3 ./scripts/build_libraries.sh   # also emit zips
#
# The version you pass MUST equal the app's expectedVersion for that pack in
# sas-app/src/shared/constants/sample-packs.ts (plain string match), or the app
# treats the installed library as a different/stale version. Bump both together.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${SAS_OUTPUTS_DIR:-}" && -f /workspace/.bash_env ]]; then
  # shellcheck disable=SC1091
  source /workspace/.bash_env
fi

OUT="${OUT:-${REPO_ROOT}/dist}"
FMT="${FMT:-dir}"   # dir | zip | both
: "${DRUM_VERSION:?set DRUM_VERSION (== sas-app DRUM_PACK.expectedVersion)}"
: "${INSTRUMENT_VERSION:?set INSTRUMENT_VERSION (== sas-app INSTRUMENT_PACK.expectedVersion)}"

echo "[build_libraries] drums v${DRUM_VERSION} + instruments v${INSTRUMENT_VERSION} (format=${FMT}) -> ${OUT}"
python3 scripts/build_pack.py --pack drums       --version "${DRUM_VERSION}"       --format "${FMT}" --out "${OUT}"
python3 scripts/build_pack.py --pack instruments --version "${INSTRUMENT_VERSION}" --format "${FMT}" --out "${OUT}"

echo
echo "[build_libraries] done."
if [[ "${FMT}" == "dir" || "${FMT}" == "both" ]]; then
  echo "Ready-to-consume libraries (the app reads <userData>/sample-packs/<subdir>/):"
  echo "  rsync -a '${OUT}/drums/'        '<userData>/sample-packs/drums/'"
  echo "  rsync -a '${OUT}/instruments/'  '<userData>/sample-packs/instruments/'"
  echo "  (<userData> on macOS = ~/Library/Application Support/signals-and-sorcery)"
fi
