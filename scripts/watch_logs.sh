#!/usr/bin/env bash
# =============================================================================
# watch_logs.sh — Live-tail run_all.log from remote via SSH.
# Run from LOCAL PC (WSL) while training is in progress.
#
# Usage:
#   bash scripts/watch_logs.sh <user> <remote_ip> [remote_project_path]
#
# Press Ctrl+C to stop watching.
# =============================================================================

REMOTE_USER="${1:-ubuntu}"
REMOTE_IP="${2:?Usage: $0 <user> <remote_ip> [remote_path]}"
REMOTE_PATH="${3:-~/hateful_memes}"

echo "=== Watching run_all.log on ${REMOTE_USER}@${REMOTE_IP} ==="
echo "=== Press Ctrl+C to stop ==="
echo ""

ssh "${REMOTE_USER}@${REMOTE_IP}" "tail -f ${REMOTE_PATH}/run_all.log"
