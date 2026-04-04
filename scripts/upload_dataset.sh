#!/usr/bin/env bash
# =============================================================================
# upload_dataset.sh — Upload Hateful Memes Dataset to remote machine.
# Run from LOCAL PC (WSL).
#
# Usage:
#   bash scripts/upload_dataset.sh <user> <remote_ip> [remote_project_path]
#
# Example:
#   bash scripts/upload_dataset.sh ubuntu 192.168.1.100 ~/hateful_memes
#
# The script expects your local dataset at the path configured in LOCAL_DATA_DIR.
# Edit that variable below to match your actual dataset location.
# =============================================================================
set -euo pipefail

REMOTE_USER="${1:-ubuntu}"
REMOTE_IP="${2:?Usage: $0 <user> <remote_ip> [remote_path]}"
REMOTE_PATH="${3:-~/hateful_memes/data}"

# ── Edit this to point to your local dataset ─────────────────────────────────
LOCAL_DATA_DIR="/mnt/d/vsu/hateful_memes_data"
# On Windows the D: drive is at /mnt/d/ in WSL
# So D:\Users\you\hateful_memes_data → /mnt/d/Users/you/hateful_memes_data
# ─────────────────────────────────────────────────────────────────────────────

if [[ ! -d "$LOCAL_DATA_DIR" ]]; then
    echo "ERROR: Dataset directory not found: $LOCAL_DATA_DIR"
    echo "Edit LOCAL_DATA_DIR in this script to match your dataset location."
    exit 1
fi

echo "=== Uploading dataset ==="
echo "  From : $LOCAL_DATA_DIR"
echo "  To   : ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
echo "  Size : $(du -sh "$LOCAL_DATA_DIR" | cut -f1)"
echo ""

# rsync: archive mode, verbose, progress, compress, resume interrupted transfers
rsync -avz --progress --partial \
    "$LOCAL_DATA_DIR/" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}/"

echo ""
echo "=== Dataset upload complete ==="
