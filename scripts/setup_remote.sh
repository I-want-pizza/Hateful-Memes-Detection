#!/usr/bin/env bash
# =============================================================================
# setup_remote.sh — Run ONCE on the remote machine after SSH login.
#
# Usage:
#   bash scripts/setup_remote.sh
# =============================================================================
set -euo pipefail

echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    transformers \
    peft \
    wandb \
    scikit-learn \
    Pillow \
    tqdm \
    pyyaml \
    numpy

echo ""
echo "=== Checking GPU ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "=== Login to wandb (paste your API key when prompted) ==="
wandb login

echo ""
echo "=== Creating data/ directory ==="
mkdir -p data/img

echo ""
echo "=== Setup complete! ==="
echo "Next step: upload dataset from your local PC:"
echo "  bash scripts/upload_dataset.sh <REMOTE_USER> <REMOTE_IP> <REMOTE_PATH>"
