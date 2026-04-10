#!/usr/bin/env bash
# =============================================================================
# watch_logs.sh — Live-tail run_all.log from remote via SSH.
# Run from LOCAL PC (WSL) while training is in progress.
#
# Usage:
#   bash scripts/watch_logs.sh
#
# All parameters are read from .env (via scripts/load_env.sh):
#   REMOTE_USER   — SSH user  (default: root)
#   REMOTE_IP     — SSH host  (REQUIRED)
#   REMOTE_PORT   — SSH port  (default: 22)
#   REMOTE_PATH   — remote project root (default: ~/hateful_memes)
#
# Press Ctrl+C to stop watching.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"

if [[ -z "${REMOTE_IP:-}" ]]; then
    echo "ERROR: REMOTE_IP is not set. Set it in .env"
    exit 1
fi

PORT="${REMOTE_PORT:-22}"
RPATH="${REMOTE_PATH:-~/hateful_memes}"

echo "=== Watching run_all.log on ${REMOTE_USER}@${REMOTE_IP}:${PORT} ==="
echo "=== Press Ctrl+C to stop ==="
echo ""

ssh -p "${PORT}" "${REMOTE_USER}@${REMOTE_IP}" "tail -f ${RPATH}/run_all.log"
