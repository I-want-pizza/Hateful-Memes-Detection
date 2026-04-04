#!/usr/bin/env bash
# =============================================================================
# download_results.sh — Pull all training outputs from remote to local PC.
# Run from LOCAL PC (WSL) after training is finished (or at any point to sync).
#
# Usage:
#   bash scripts/download_results.sh <user> <remote_ip> [remote_project_path]
#
# Downloads:
#   outputs/        — checkpoints, logs, metrics CSVs, eval JSONs
#   results_summary.csv
#   ablation_results.csv
#   run_all.log
# Does NOT download:
#   data/           — 3-4 GB, no need to bring back
# =============================================================================
set -euo pipefail

REMOTE_USER="${1:-ubuntu}"
REMOTE_IP="${2:?Usage: $0 <user> <remote_ip> [remote_path]}"
REMOTE_PATH="${3:-~/hateful_memes}"

LOCAL_DEST="/mnt/d/vsu/8th_semester/diplom/results"
mkdir -p "$LOCAL_DEST"

echo "=== Downloading results from ${REMOTE_USER}@${REMOTE_IP} ==="
echo "  Remote: ${REMOTE_PATH}"
echo "  Local : ${LOCAL_DEST}"
echo ""

rsync -avz --progress --partial \
    --exclude="data/" \
    --exclude="*.pyc" \
    --exclude="__pycache__/" \
    --include="outputs/***" \
    --include="results_summary.csv" \
    --include="ablation_results.csv" \
    --include="run_all.log" \
    --include="*/" \
    --exclude="*" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}/" \
    "$LOCAL_DEST/"

echo ""
echo "=== Download complete → $LOCAL_DEST ==="
