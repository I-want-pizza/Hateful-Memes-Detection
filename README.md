# Hateful Memes Classification — Experiment Pipeline

Multimodal hate speech detection via CLIP (late fusion) and ViLT (early fusion).
21 experiments: encoder strategies × fusion methods × regularization.
Dataset: [Hateful Memes](https://ai.meta.com/tools/hatefulmemes/) (Meta AI Research)

## Project structure

```
code/
├── configs/                    # YAML config for each of 21 experiments
├── config/settings.py          # Pydantic settings — reads from .env
├── dataset/hateful_memes.py
├── models/
│   ├── clip_classifier.py
│   └── vilt_classifier.py
├── trainers/trainer.py         # wandb + CSV logging, FP16, gradient accum
├── evaluators/evaluator.py     # ablation-mode evaluation
├── utils/                      # metrics, collate, checkpoint, logger
├── analysis/
│   ├── plots.py                # figs 5-9: learning curves, param efficiency, fusion
│   ├── ablation_heatmap.py     # modality ablation heatmap
│   ├── sota_comparison.py      # comparison with published baselines
│   ├── ensemble.py             # CLIP + ViLT ensemble evaluation
│   ├── confusion_matrix_plot.py# fig 10: confusion matrices
│   ├── error_analysis.py       # hard examples (FP/FN grid)
│   ├── attention_maps.py       # CLIP attention map visualization
│   └── run_analysis.sh         # runs all 7 analysis scripts in order
├── scripts/                    # helper scripts (see table below)
├── train.py                    # train a single experiment
├── evaluate.py                 # evaluate checkpoint (any split + ablation)
├── run_ablation.py             # full modality ablation study
├── summarize_results.py        # compile all results → CSV table
├── test_smoke.py               # smoke test before full run
├── run_all.sh                  # master script: trains + ablation + analysis
├── requirements.txt
├── .env.example                # config template
└── .env                        # your secrets — never commit
```

## Setup

### 1. Configure `.env`

```bash
cp .env.example .env
# fill in: REMOTE_IP, REMOTE_PORT, WANDB_API_KEY, DATASET_GDRIVE_ID
```

### 2. Upload dataset to Yandex Disk (once)

1. Pack your local dataset folder into an archive:
   ```powershell
   tar -czf "$env:TEMP\hateful_memes_data.zip" -C "D:\path\to" data
   ```
   (or zip via Explorer — whatever format you prefer: `.zip`, `.tar.gz`)
2. Upload the archive to Yandex Disk (browser or desktop app)
3. Note the path, e.g. `disk:/hateful_memes/data.zip`
4. Set in `.env`: `YADISK_DATASET_PATH=disk:/hateful_memes/data.zip`

### 3. Upload code to remote (Windows PowerShell)

```powershell
cd D:\path\to\diplom\code
powershell -ExecutionPolicy Bypass -File scripts\upload_code.ps1
```

Also copy `.env` to remote:
```powershell
scp -P $REMOTE_PORT .env root@$REMOTE_IP:~/hateful_memes/.env
```

### 4. Set up remote machine (run once on server)

```bash
ssh -p <PORT> root@<IP>
cd ~/hateful_memes  
bash scripts/setup_remote.sh
```

### 5. Download dataset on remote

```bash
bash scripts/download_dataset_remote.sh
```

### 6. Run smoke test (optional but recommended)

```bash
bash scripts/run_smoke_test.sh
```

### 7. Run all 21 experiments + analysis

```bash
# vast.ai auto-starts tmux — you're already inside a session
bash run_all.sh
```

`run_all.sh` trains all 21 configs sequentially, skipping any that have a `best.pt` checkpoint already.
After training it automatically runs: ablation study → results summary → all analysis scripts → figures.

Detach with `Ctrl+B, D` — training continues in background.

### 8. Monitor from local machine

```powershell
powershell -ExecutionPolicy Bypass -File scripts\watch_logs.ps1
```

Or open **Weights & Biases**: https://wandb.ai

### 9. Download results when done

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_results.ps1
```

Downloads `outputs/` (metrics CSVs, eval JSONs — no `.pt` checkpoints), `results_summary.csv`,
`ablation_results.csv`, and `analysis_outputs/` (all figures).
Checkpoints are uploaded to Yandex Disk separately (see step below).

### 10. (Optional) Upload checkpoints only to Yandex Disk

```bash
# on remote, after training — fast, only best.pt files + CSVs
bash scripts/upload_checkpoints_yadisk.sh
```

### 11. (Optional) Full backup to Yandex Disk

```bash
# on remote — packs EVERYTHING: data/ + outputs/ + analysis_outputs/ + code
# splits into BACKUP_CHUNK_GB parts (default 5 GB) and uploads
bash scripts/backup_full_yadisk.sh
```

> `run_all.sh` calls this automatically at the very end.

## Scripts reference

| Script | Where to run | What it does |
|---|---|---|
| `scripts/upload_code.ps1` | local PowerShell | Pack & upload code to remote via scp |
| `scripts/watch_logs.ps1` | local PowerShell | Live-tail `run_all.log` via SSH |
| `scripts/download_results.ps1` | local PowerShell | Download outputs + figures from remote |
| `scripts/setup_remote.sh` | remote bash | Install deps via `requirements.txt` + wandb login |
| `scripts/download_dataset_remote.sh` | remote bash | Download dataset from Yandex Disk via REST API |
| `scripts/run_smoke_test.sh` | remote bash | Run smoke test before full training |
| `scripts/upload_checkpoints_yadisk.sh` | remote bash | Upload `best.pt` + CSVs to Yandex Disk |
| `scripts/backup_full_yadisk.sh` | remote bash | Full backup (data+outputs+code) → Yandex Disk |
| `scripts/load_env.sh` | sourced by remote scripts | Parse `.env` into bash variables |
| `scripts/load_env.ps1` | sourced by PS scripts | Parse `.env` into PS variables |
| `analysis/run_analysis.sh` | remote bash | All 7 analysis scripts → `analysis_outputs/` |

## Single experiment

```bash
python train.py --config clip_lora_r8
```

## Evaluation & ablation

```bash
# Evaluate checkpoint on test set
python evaluate.py --config clip_lora_r8 --split test_seen

# Text-only modality ablation on dev_seen
python evaluate.py --config clip_lora_r8 --split dev_seen --ablation text_only

# Full ablation study (best CLIP + best ViLT × all splits × all modes)
python run_ablation.py

# Compile all results into one CSV table
python summarize_results.py
```

## Experiments overview

| ID | Config | Model | Encoder strategy | Fusion |
|---|---|---|---|---|
| B1 | `clip_frozen_concat` | CLIP | Frozen | Concat |
| B2 | `vilt_full_ft` | ViLT | Full FT | Early |
| C2–C5 | `clip_lora_r{4,8,16,32}` | CLIP | LoRA | Concat |
| C6–C8 | `clip_unfreeze_{1,3,6}` | CLIP | Partial unfreeze | Concat |
| F2 | `clip_frozen_elemwise` | CLIP | Frozen | Element-wise |
| F3 | `clip_frozen_gated` | CLIP | Frozen | Gated |
| F4 | `clip_frozen_xattn1` | CLIP | Frozen | CrossAttn (1L) |
| F5 | `clip_frozen_xattn2` | CLIP | Frozen | CrossAttn (2L) |
| CF | `clip_lora_r8_xattn1` | CLIP | LoRA r=8 | CrossAttn (1L) |
| V2–V4 | `vilt_lora_r{4,8,16}` | ViLT | LoRA | Early |
| V5–V6 | `vilt_unfreeze_{3,6}` | ViLT | Partial unfreeze | Early |
| V7 | `vilt_label_smooth` | ViLT | Full FT | Early + label smooth |
| V8 | `vilt_dropout_smooth` | ViLT | Full FT | Early + dropout |

## Requirements

- Python 3.10+
- CUDA GPU (tested on RTX 5080)
- See `requirements.txt` for Python packages
