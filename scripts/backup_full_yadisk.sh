#!/usr/bin/env bash
# =============================================================================
# backup_full_yadisk.sh — Full project backup to Yandex Disk.
#
# Packs the ENTIRE working directory (code + data + outputs + checkpoints)
# into a timestamped archive, splits if > CHUNK_GB, and uploads to Yandex Disk.
#
# Run ON THE SERVER after training + analysis is complete:
#   bash scripts/backup_full_yadisk.sh
#
# Reads from .env:
#   YADISK_TOKEN   — OAuth token
#   YADISK_PATH    — root folder on Yandex Disk (default: hateful_memes)
#   REMOTE_PATH    — local project root       (default: ~/hateful_memes)
#
# Yandex Disk REST API limit: 50 GB per file.
# Script splits at CHUNK_GB (default: 5 GB) just to be safe with slow networks.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Load .env ─────────────────────────────────────────────────────────────────
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${YADISK_TOKEN:-}" ]]; then
    echo "ERROR: YADISK_TOKEN not set in .env"
    exit 1
fi

YADISK_PATH="${YADISK_PATH:-hateful_memes}"
BASE_URL="https://cloud-api.yandex.net/v1/disk/resources"
AUTH="Authorization: OAuth ${YADISK_TOKEN}"
CHUNK_GB="${BACKUP_CHUNK_GB:-5}"          # split threshold in GB
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="hateful_memes_full_${TIMESTAMP}"
ARCHIVE_DIR="/tmp/yadisk_backup_${TIMESTAMP}"
mkdir -p "$ARCHIVE_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

yadisk_mkdir() {
    curl -s -o /dev/null -w "%{http_code}" -X PUT \
        -H "$AUTH" \
        "${BASE_URL}?path=${1}" || true
}

yadisk_upload_file() {
    local local_path="$1"
    local remote_path="$2"
    local size_mb
    size_mb=$(du -m "$local_path" | cut -f1)

    log "  Uploading $(basename "$local_path") (${size_mb} MB) -> disk:/${remote_path}"

    local upload_url
    upload_url=$(curl -sf \
        -H "$AUTH" \
        "${BASE_URL}/upload?path=${remote_path}&overwrite=true" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['href'])")

    curl -f --progress-bar \
        -T "$local_path" \
        "$upload_url"

    log "  Done: $(basename "$local_path")"
}

# ── Step 1: Pack ──────────────────────────────────────────────────────────────
log "=== Full backup: packing $PROJECT_DIR ==="

ARCHIVE_TAR="${ARCHIVE_DIR}/${ARCHIVE_NAME}.tar.gz"

tar -czf "$ARCHIVE_TAR" \
    --exclude="$PROJECT_DIR/wandb" \
    --exclude="$PROJECT_DIR/.git" \
    --exclude="$PROJECT_DIR/__pycache__" \
    --exclude="$PROJECT_DIR/*/__pycache__" \
    --exclude="$PROJECT_DIR/*.pyc" \
    --exclude="$PROJECT_DIR/outputs/*/wandb" \
    -C "$(dirname "$PROJECT_DIR")" \
    "$(basename "$PROJECT_DIR")"

ARCHIVE_SIZE_MB=$(du -m "$ARCHIVE_TAR" | cut -f1)
log "Archive: $ARCHIVE_TAR (${ARCHIVE_SIZE_MB} MB)"

# ── Step 2: Split if needed ───────────────────────────────────────────────────
CHUNK_MB=$(( CHUNK_GB * 1024 ))

if [[ $ARCHIVE_SIZE_MB -gt $CHUNK_MB ]]; then
    log "Archive > ${CHUNK_GB} GB — splitting into ${CHUNK_GB} GB chunks..."
    split -b "${CHUNK_MB}m" "$ARCHIVE_TAR" "${ARCHIVE_DIR}/${ARCHIVE_NAME}.tar.gz.part_"
    rm "$ARCHIVE_TAR"
    PARTS=( "${ARCHIVE_DIR}/${ARCHIVE_NAME}.tar.gz.part_"* )
    log "Split into ${#PARTS[@]} parts"
else
    PARTS=( "$ARCHIVE_TAR" )
fi

# ── Step 3: Upload ────────────────────────────────────────────────────────────
log ""
log "=== Uploading to Yandex Disk: disk:/${YADISK_PATH}/backups/ ==="

yadisk_mkdir "disk:/${YADISK_PATH}"
yadisk_mkdir "disk:/${YADISK_PATH}/backups"
yadisk_mkdir "disk:/${YADISK_PATH}/backups/${ARCHIVE_NAME}"

for part in "${PARTS[@]}"; do
    filename=$(basename "$part")
    yadisk_upload_file "$part" "${YADISK_PATH}/backups/${ARCHIVE_NAME}/${filename}"
done

# Also save a short manifest so we know what's in this backup
MANIFEST="${ARCHIVE_DIR}/manifest.txt"
cat > "$MANIFEST" <<EOF
Backup: ${ARCHIVE_NAME}
Date:   $(date)
Host:   $(hostname)
Source: $PROJECT_DIR
Parts:  ${#PARTS[@]}
Size:   ${ARCHIVE_SIZE_MB} MB

Contents:
$(du -sh "$PROJECT_DIR"/data        2>/dev/null && echo "  data/")
$(du -sh "$PROJECT_DIR"/outputs     2>/dev/null && echo "  outputs/")
$(du -sh "$PROJECT_DIR"/analysis_outputs 2>/dev/null && echo "  analysis_outputs/")
Experiments trained: $(ls "$PROJECT_DIR"/outputs/*/best.pt 2>/dev/null | wc -l) / 21
EOF

yadisk_upload_file "$MANIFEST" "${YADISK_PATH}/backups/${ARCHIVE_NAME}/manifest.txt"

# ── Cleanup ───────────────────────────────────────────────────────────────────
rm -rf "$ARCHIVE_DIR"

log ""
log "=== Backup complete! ==="
log "  Location : https://disk.yandex.ru  →  ${YADISK_PATH}/backups/${ARCHIVE_NAME}/"
log "  Parts    : ${#PARTS[@]}"
log "  Size     : ${ARCHIVE_SIZE_MB} MB"
log ""
log "To restore on a new machine:"
log "  # Download all parts, then:"
if [[ ${#PARTS[@]} -gt 1 ]]; then
log "  cat ${ARCHIVE_NAME}.tar.gz.part_* > ${ARCHIVE_NAME}.tar.gz"
fi
log "  tar -xzf ${ARCHIVE_NAME}.tar.gz -C /root/"
