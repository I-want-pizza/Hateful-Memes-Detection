#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Master training script.
# Run inside tmux on the remote machine:
#   tmux new -s training
#   bash run_all.sh
# =============================================================================
set -euo pipefail

LOG_FILE="run_all.log"
PYTHON="${PYTHON:-python}"   # override with: PYTHON=python3 bash run_all.sh

# ── All experiments in execution order ───────────────────────────────────────
CONFIGS=(
    # Group 1: Baselines
    "clip_frozen_concat"
    "vilt_full_ft"

    # Group 2: CLIP encoder strategies
    "clip_lora_r4"
    "clip_lora_r8"
    "clip_lora_r16"
    "clip_lora_r32"
    "clip_unfreeze_1"
    "clip_unfreeze_3"
    "clip_unfreeze_6"

    # Group 3: CLIP fusion methods (frozen encoder)
    "clip_frozen_elemwise"
    "clip_frozen_gated"
    "clip_frozen_xattn1"
    "clip_frozen_xattn2"

    # CF-BEST: best encoder + best fusion
    "clip_lora_r8_xattn1"

    # Group 4: ViLT strategies
    "vilt_lora_r4"
    "vilt_lora_r8"
    "vilt_lora_r16"
    "vilt_unfreeze_3"
    "vilt_unfreeze_6"
    "vilt_label_smooth"
    "vilt_dropout_smooth"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "====== Starting run_all.sh ======"
log "Total experiments: ${#CONFIGS[@]}"
log "Python: $($PYTHON --version 2>&1)"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# ── Main loop ─────────────────────────────────────────────────────────────────
TOTAL=${#CONFIGS[@]}
DONE=0
SKIPPED=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    CKPT="outputs/${CONFIG}/best.pt"

    if [[ -f "$CKPT" ]]; then
        log "SKIP [$((DONE+SKIPPED+FAILED+1))/$TOTAL] $CONFIG (checkpoint exists)"
        ((SKIPPED++)) || true
        continue
    fi

    log "START [$((DONE+SKIPPED+FAILED+1))/$TOTAL] $CONFIG"
    START_TS=$(date +%s)

    # Create output dir and capture stdout+stderr
    mkdir -p "outputs/${CONFIG}"

    set +e
    $PYTHON train.py --config "$CONFIG" 2>&1 | tee "outputs/${CONFIG}/stdout.log"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [[ $EXIT_CODE -eq 0 ]]; then
        log "DONE  $CONFIG in ${ELAPSED}s"
        ((DONE++)) || true
    else
        log "FAIL  $CONFIG (exit code $EXIT_CODE) — continuing to next"
        ((FAILED++)) || true
    fi
done

log "====== All experiments finished: $DONE done, $SKIPPED skipped, $FAILED failed ======"

# ── Post-training analysis ────────────────────────────────────────────────────
log "Running ablation study..."
$PYTHON run_ablation.py 2>&1 | tee -a "$LOG_FILE" || log "Ablation failed (non-fatal)"

log "Compiling results summary..."
$PYTHON summarize_results.py 2>&1 | tee -a "$LOG_FILE" || log "Summary failed (non-fatal)"

log "====== COMPLETE — see results_summary.csv and ablation_results.csv ======"
