#!/usr/bin/env bash
# Move outputs between the pod (volume) and a cheap object store (Backblaze B2,
# Cloudflare R2, S3, …) via rclone.
#
# One-time config on a fresh pod (interactive, ~2 minutes):
#   pip install rclone || apt-get install -y rclone
#   rclone config            # create a remote named "samples" pointing at B2/R2
#
# Then call this script:
#   ./sync.sh push                       # outputs/ -> samples:sas-samples/<host>/
#   ./sync.sh push outputs/processed     # only the processed subdir
#   ./sync.sh pull                       # samples:sas-samples/<host>/ -> outputs/
#   ./sync.sh ls                         # list what's on the remote
#
# Override the remote name or bucket path via env:
#   RCLONE_REMOTE=mybucket RCLONE_BUCKET=sas-samples ./sync.sh push

set -euo pipefail

RCLONE_REMOTE="${RCLONE_REMOTE:-samples}"
RCLONE_BUCKET="${RCLONE_BUCKET:-sas-samples}"
RUN_TAG="${RUN_TAG:-$(hostname)}"
LOCAL_DIR="${LOCAL_DIR:-${SAS_OUTPUTS_DIR:-outputs}}"
REMOTE_PATH="${RCLONE_REMOTE}:${RCLONE_BUCKET}/${RUN_TAG}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "[sync] rclone not installed. Install with: apt-get install -y rclone" >&2
  exit 1
fi

if ! rclone listremotes | grep -q "^${RCLONE_REMOTE}:"; then
  echo "[sync] no rclone remote named '${RCLONE_REMOTE}'. Run 'rclone config' first." >&2
  exit 1
fi

cmd="${1:-push}"
subdir="${2:-}"

case "${cmd}" in
  push)
    src="${LOCAL_DIR}${subdir:+/${subdir}}"
    dst="${REMOTE_PATH}${subdir:+/${subdir}}"
    echo "[sync] push ${src} -> ${dst}"
    rclone copy --progress --transfers 8 --checkers 16 "${src}" "${dst}"
    ;;
  pull)
    src="${REMOTE_PATH}${subdir:+/${subdir}}"
    dst="${LOCAL_DIR}${subdir:+/${subdir}}"
    echo "[sync] pull ${src} -> ${dst}"
    mkdir -p "${dst}"
    rclone copy --progress --transfers 8 --checkers 16 "${src}" "${dst}"
    ;;
  ls)
    echo "[sync] listing ${REMOTE_PATH}"
    rclone lsd "${REMOTE_PATH}" 2>/dev/null || echo "[sync] (empty or missing)"
    ;;
  size)
    echo "[sync] size of ${REMOTE_PATH}"
    rclone size "${REMOTE_PATH}"
    ;;
  *)
    echo "usage: $0 {push|pull|ls|size} [subdir]" >&2
    exit 2
    ;;
esac
