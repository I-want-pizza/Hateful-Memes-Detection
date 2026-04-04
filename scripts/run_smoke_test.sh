#!/usr/bin/env bash
# =============================================================================
# run_smoke_test.sh — Запустить smoke test локально (WSL) или на remote.
#
# Usage (из папки code/):
#   bash scripts/run_smoke_test.sh /path/to/dataset
#   bash scripts/run_smoke_test.sh /mnt/d/path/to/dataset  # WSL Windows путь
#   bash scripts/run_smoke_test.sh /mnt/d/path/to/dataset --skip-vilt  # только CLIP
# =============================================================================

DATA_DIR="${1:?Usage: $0 <path-to-dataset> [--skip-vilt]}"
EXTRA_ARGS="${2:-}"

echo "=== Установка зависимостей (если нужно) ==="
pip install -q torch torchvision transformers peft wandb scikit-learn Pillow tqdm pyyaml

echo ""
echo "=== Запуск smoke test ==="
python test_smoke.py --data-dir "$DATA_DIR" $EXTRA_ARGS
