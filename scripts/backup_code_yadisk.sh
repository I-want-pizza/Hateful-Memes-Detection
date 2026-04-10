#!/usr/bin/env bash
# =============================================================================
# backup_code_yadisk.sh — Lightweight code-only snapshot to Yandex Disk.
#
# Excludes data/, outputs/, analysis_outputs/, wandb/, .git/
# Typical size: 1-3 MB. Runs in seconds.
#
# Called automatically at the START of run_all.sh so code state is saved
# before a potentially multi-day training run begins.
#
# Run ON THE SERVER from ~/hateful_memes/:
#   bash scripts/backup_code_yadisk.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${YADISK_TOKEN:-}" ]]; then
    echo "ERROR: YADISK_TOKEN not set in .env"
    exit 1
fi

YADISK_PATH="${YADISK_PATH:-hateful_memes}"
BASE_URL="https://cloud-api.yandex.net/v1/disk/resources"
AUTH="Authorization: OAuth ${YADISK_TOKEN}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE="/tmp/hateful_memes_code_${TIMESTAMP}.tar.gz"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

yadisk_mkdir() {
    curl -s -o /dev/null -X PUT -H "$AUTH" "${BASE_URL}?path=${1}" || true
}

yadisk_upload_file() {
    local local_path="$1"
    local remote_path="$2"
    local upload_url
    upload_url=$(curl -sf \
        -H "$AUTH" \
        "${BASE_URL}/upload?path=${remote_path}&overwrite=true" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['href'])")
    curl -f --progress-bar -T "$local_path" "$upload_url"
}

log "=== Code snapshot → Yandex Disk ==="

tar -czf "$ARCHIVE" \
    --exclude="$PROJECT_DIR/data" \
    --exclude="$PROJECT_DIR/outputs" \
    --exclude="$PROJECT_DIR/analysis_outputs" \
    --exclude="$PROJECT_DIR/wandb" \
    --exclude="$PROJECT_DIR/.git" \
    --exclude="$PROJECT_DIR/__pycache__" \
    --exclude="$PROJECT_DIR/*/__pycache__" \
    --exclude="$PROJECT_DIR/*.pyc" \
    --exclude="$PROJECT_DIR/.env" \
    -C "$(dirname "$PROJECT_DIR")" \
    "$(basename "$PROJECT_DIR")"

SIZE_MB=$(du -m "$ARCHIVE" | cut -f1)
log "Archive: ${SIZE_MB} MB"

yadisk_mkdir "disk:/${YADISK_PATH}"
yadisk_mkdir "disk:/${YADISK_PATH}/code_snapshots"

REMOTE_NAME="code_${TIMESTAMP}.tar.gz"
yadisk_upload_file "$ARCHIVE" "${YADISK_PATH}/code_snapshots/${REMOTE_NAME}"

rm -f "$ARCHIVE"

log "Done: disk:/${YADISK_PATH}/code_snapshots/${REMOTE_NAME}"
