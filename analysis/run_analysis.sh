#!/usr/bin/env bash
# =============================================================================
# run_analysis.sh — Run all analysis scripts in order.
# Run ON THE SERVER from ~/hateful_memes/:
#   bash analysis/run_analysis.sh
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p analysis_outputs analysis_outputs/plots

echo "[1/7] Training curves and result plots (figs 5-9)..."
python analysis/plots.py

echo ""
echo "[2/7] Ablation heatmaps..."
python analysis/ablation_heatmap.py

echo ""
echo "[3/7] SOTA comparison..."
python analysis/sota_comparison.py

echo ""
echo "[4/7] Ensemble evaluation..."
python analysis/ensemble.py

echo ""
echo "[5/7] Confusion matrices (fig 10)..."
python analysis/confusion_matrix_plot.py

echo ""
echo "[6/7] Error analysis..."
python analysis/error_analysis.py

echo ""
echo "[7/7] Attention maps..."
python analysis/attention_maps.py

echo ""
echo "=== All analysis done ==="
echo "Results in: analysis_outputs/"
ls -lh analysis_outputs/ analysis_outputs/plots/ 2>/dev/null || ls -lh analysis_outputs/
