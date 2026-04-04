# Hateful Memes Classification

Multimodal meme classification experiments for bachelor's thesis.
Dataset: [Hateful Memes Dataset](https://ai.meta.com/tools/hatefulmemes/) (Facebook AI Research)

## Project structure

```
├── configs/          # One YAML per experiment (21 total)
├── dataset/          # HatefulMemesDataset with per-split support
├── models/           # CLIPClassifier, ViLTClassifier
├── trainers/         # Trainer with wandb + CSV logging
├── evaluators/       # Evaluator with ablation-mode support
├── utils/            # metrics, collate, checkpoint, logger
├── scripts/          # setup_remote, upload_dataset, download_results, watch_logs
├── train.py          # Train a single experiment
├── evaluate.py       # Evaluate a checkpoint (any split + ablation mode)
├── run_ablation.py   # Full modality ablation study
├── summarize_results.py  # Compile all results → CSV
└── run_all.sh        # Master script: runs all 21 experiments
```

## Quick start (remote machine)

```bash
# 1. Clone repo
git clone <your-repo-url> && cd hateful_memes

# 2. Install dependencies + login to wandb
bash scripts/setup_remote.sh

# 3. Upload dataset from local PC (run on LOCAL WSL):
bash scripts/upload_dataset.sh ubuntu <REMOTE_IP> ~/hateful_memes/data

# 4. On remote — start all experiments in tmux
tmux new -s training
bash run_all.sh
```

## Monitoring

**wandb (browser):** https://wandb.ai — real-time metrics, loss curves, GPU stats

**Live log (local WSL):**
```bash
bash scripts/watch_logs.sh ubuntu <REMOTE_IP>
```

## Single experiment

```bash
python train.py --config clip_lora_r8
```

## Evaluation & ablation

```bash
# Evaluate on test set
python evaluate.py --config clip_lora_r8 --split test

# Text-only ablation on dev_seen
python evaluate.py --config clip_lora_r8 --split dev_seen --ablation text_only

# Full ablation study (best CLIP + best ViLT on all splits × all modes)
python run_ablation.py

# Compile all results into one table
python summarize_results.py
```

## Download results (local WSL)

```bash
bash scripts/download_results.sh ubuntu <REMOTE_IP>
```

## Experiments

| ID  | Config                  | Encoder    | Fusion     |
|-----|-------------------------|------------|------------|
| B1  | clip_frozen_concat      | Frozen     | Concat     |
| B2  | vilt_full_ft            | Full FT    | Early      |
| C2-C5 | clip_lora_r{4,8,16,32} | LoRA     | Concat     |
| C6-C8 | clip_unfreeze_{1,3,6}  | Unfreeze N | Concat     |
| F2  | clip_frozen_elemwise    | Frozen     | Elem-wise  |
| F3  | clip_frozen_gated       | Frozen     | Gated      |
| F4  | clip_frozen_xattn1      | Frozen     | CrossAttn1 |
| F5  | clip_frozen_xattn2      | Frozen     | CrossAttn2 |
| CF  | clip_lora_r8_xattn1     | LoRA r=8   | CrossAttn1 |
| V2-V4 | vilt_lora_r{4,8,16}  | LoRA       | Early      |
| V5-V6 | vilt_unfreeze_{3,6}  | Unfreeze N | Early      |
| V7  | vilt_label_smooth       | Full FT    | Early + LS |
| V8  | vilt_dropout_smooth     | Full FT    | Early + Reg|
