#!/usr/bin/env bash
# =============================================================================
# setup_remote.sh — Run ONCE on the remote machine after SSH login.
#
# Usage (on remote, from ~/hateful_memes/):
#   bash scripts/setup_remote.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Installing Python dependencies ==="
pip install --upgrade pip

# torch/torchvision: skip if already installed (vast.ai images ship with CUDA build)
python -c "import torch" 2>/dev/null && echo "torch already installed, skipping" || \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining deps from requirements.txt (skips torch if already present)
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "=== Checking GPU ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "=== Login to wandb ==="
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/load_env.sh"
fi
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY"
    echo "  wandb logged in via .env key"
else
    wandb login   # interactive prompt
fi

echo ""
echo "=== Creating data/ directory ==="
mkdir -p "$PROJECT_DIR/data/img"
mkdir -p "$PROJECT_DIR/outputs"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download dataset:  bash scripts/download_dataset_remote.sh"
echo "  2. Smoke test:        bash scripts/run_smoke_test.sh"
echo "  3. Train all:         bash run_all.sh"
