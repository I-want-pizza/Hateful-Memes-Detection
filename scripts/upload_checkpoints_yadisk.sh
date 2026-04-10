#!/usr/bin/env bash
# =============================================================================
# upload_checkpoints_yadisk.sh — Upload best.pt checkpoints to Yandex Disk.
# Run ON THE SERVER after training is complete:
#   bash scripts/upload_checkpoints_yadisk.sh
#
# Reads from .env:
#   YADISK_TOKEN     — OAuth token from oauth.yandex.ru
#   YADISK_PATH      — remote folder on Yandex Disk (default: hateful_memes)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load .env
ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "ERROR: .env not found at $ENV_FILE"
    exit 1
fi
while IFS='=' read -r key value; do
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    export "$key"="$value"
done < "$ENV_FILE"

if [[ -z "${YADISK_TOKEN:-}" ]]; then
    echo "ERROR: YADISK_TOKEN not set in .env"
    echo "See scripts/upload_checkpoints_yadisk.sh header for instructions."
    exit 1
fi

YADISK_PATH="${YADISK_PATH:-hateful_memes}"
BASE_URL="https://cloud-api.yandex.net/v1/disk/resources"
AUTH="Authorization: OAuth ${YADISK_TOKEN}"

# Helper: create folder on Yandex Disk (ignore if exists)
yadisk_mkdir() {
    curl -s -o /dev/null -X PUT \
        -H "$AUTH" \
        "${BASE_URL}?path=${1}"
}

# Helper: upload a file
yadisk_upload() {
    local local_path="$1"
    local remote_path="$2"
    local filename
    filename="$(basename "$local_path")"

    echo "  -> $remote_path"

    # Get upload URL
    local upload_url
    upload_url=$(curl -s \
        -H "$AUTH" \
        "${BASE_URL}/upload?path=${remote_path}&overwrite=true" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['href'])")

    # Upload file
    curl -s -o /dev/null -T "$local_path" "$upload_url"
}

echo "=== Uploading checkpoints to Yandex Disk ==="
echo "  Folder : disk:/${YADISK_PATH}"
echo ""

# Create root folder
yadisk_mkdir "disk:/${YADISK_PATH}"
yadisk_mkdir "disk:/${YADISK_PATH}/checkpoints"

# Upload each best.pt
count=0
for ckpt in "$PROJECT_DIR"/outputs/*/best.pt; do
    [[ -f "$ckpt" ]] || continue
    exp=$(basename "$(dirname "$ckpt")")
    yadisk_mkdir "disk:/${YADISK_PATH}/checkpoints/${exp}"
    yadisk_upload "$ckpt" "disk:/${YADISK_PATH}/checkpoints/${exp}/best.pt"
    count=$((count + 1))
done

# Also upload logs and CSVs
for f in run_all.log results_summary.csv ablation_results.csv; do
    [[ -f "$PROJECT_DIR/$f" ]] || continue
    yadisk_upload "$PROJECT_DIR/$f" "disk:/${YADISK_PATH}/${f}"
done

echo ""
echo "=== Done! Uploaded ${count} checkpoints ==="
echo "  Check: https://disk.yandex.ru"
