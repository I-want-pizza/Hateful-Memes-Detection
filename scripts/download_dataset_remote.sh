#!/usr/bin/env bash
# =============================================================================
# download_dataset_remote.sh — Download dataset from Yandex Disk on remote.
# Run ON THE SERVER:
#   bash scripts/download_dataset_remote.sh
#
# Reads from .env:
#   YADISK_TOKEN        — OAuth token
#   YADISK_DATASET_PATH — path on Yandex Disk, e.g. disk:/hateful_memes/data.zip
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${YADISK_TOKEN:-}" ]]; then
    echo "ERROR: YADISK_TOKEN not set in .env"
    exit 1
fi

if [[ -z "${YADISK_DATASET_PATH:-}" ]]; then
    echo "ERROR: YADISK_DATASET_PATH not set in .env"
    echo "  Example: YADISK_DATASET_PATH=disk:/hateful_memes/data.zip"
    exit 1
fi

AUTH="Authorization: OAuth ${YADISK_TOKEN}"
BASE_URL="https://cloud-api.yandex.net/v1/disk/resources"

FILENAME=$(basename "$YADISK_DATASET_PATH")   # e.g. data.zip or data.tar.gz
DEST="$PROJECT_DIR/$FILENAME"

echo "=== Downloading dataset from Yandex Disk ==="
echo "  Remote : $YADISK_DATASET_PATH"
echo "  Local  : $DEST"
echo ""

# Get download URL
DOWNLOAD_URL=$(curl -sf \
    -H "$AUTH" \
    "${BASE_URL}/download?path=${YADISK_DATASET_PATH}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['href'])")

if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "ERROR: Could not get download URL. Check YADISK_DATASET_PATH and YADISK_TOKEN."
    exit 1
fi

# Download
curl -f -L --progress-bar -o "$DEST" "$DOWNLOAD_URL"

echo ""
echo "=== Extracting ==="
cd "$PROJECT_DIR"

case "$FILENAME" in
    *.zip)
        apt-get install -y -q unzip 2>/dev/null || true
        unzip -q -o "$DEST" -d .
        ;;
    *.tar.gz | *.tgz)
        tar -xzf "$DEST"
        ;;
    *.tar)
        tar -xf "$DEST"
        ;;
    *)
        echo "ERROR: Unknown archive format: $FILENAME"
        exit 1
        ;;
esac

rm "$DEST"

echo ""
echo "=== Done! ==="
ls -lh data/ | head -10
