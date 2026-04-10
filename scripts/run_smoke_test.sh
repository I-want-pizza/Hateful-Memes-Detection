#!/usr/bin/env bash
# =============================================================================
# run_smoke_test.sh — Запустить smoke test локально (WSL) или на remote.
#
# Usage (из папки code/):
#   bash scripts/run_smoke_test.sh              # DATA_DIR из .env
#   bash scripts/run_smoke_test.sh --skip-vilt  # только CLIP тесты
#
# DATA_DIR можно переопределить прямо из командной строки:
#   DATA_DIR=/mnt/d/path/to/dataset bash scripts/run_smoke_test.sh
# =============================================================================
set -euo pipefail

# Load .env variables (если файл есть)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/load_env.sh" ]]; then
    source "${SCRIPT_DIR}/load_env.sh"
fi

RESOLVED_DATA_DIR="${DATA_DIR:-data}"
EXTRA_ARGS="${1:-}"

echo "=== Установка зависимостей (если нужно) ==="
pip install -q torch torchvision transformers peft wandb scikit-learn Pillow tqdm pyyaml pydantic-settings

echo ""
echo "=== Запуск smoke test ==="
echo "  DATA_DIR : $RESOLVED_DATA_DIR"
echo ""
python test_smoke.py --data-dir "$RESOLVED_DATA_DIR" $EXTRA_ARGS
