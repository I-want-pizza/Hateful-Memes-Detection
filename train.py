"""
Entry point for training a single experiment.

Usage:
    python train.py --config clip_frozen_concat
    python train.py --config vilt_lora_r8
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, ViltProcessor

from dataset.hateful_memes import HatefulMemesDataset
from models.clip_classifier import CLIPClassifier
from models.vilt_classifier import ViLTClassifier
from trainers.trainer import Trainer
from utils.collate import MultimodalCollator
from utils.logger import get_logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_optimizer(model, cfg: dict):
    tcfg     = cfg["training"]
    strategy = cfg["model"].get("encoder_strategy", "frozen")

    # Discriminative LR for partial-unfreeze experiments
    if strategy.startswith("unfreeze_"):
        lr_backbone = float(tcfg.get("lr_backbone", 2e-6))
        lr_head     = float(tcfg["lr"])
        backbone_params, head_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in ("classifier", "gate", "cross_attn")):
                head_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params,     "lr": lr_head},
        ]
    else:
        param_groups = filter(lambda p: p.requires_grad, model.parameters())

    return torch.optim.AdamW(
        param_groups,
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Config name (without .yaml) from configs/ dir")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────
    config_path = Path("configs") / f"{args.config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Coerce numeric types that YAML might load as strings
    cfg["training"]["lr"]           = float(cfg["training"]["lr"])
    cfg["training"]["weight_decay"] = float(cfg["training"].get("weight_decay", 0.0))

    set_seed(args.seed)

    out_dir = Path(cfg["experiment"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot next to outputs
    import shutil
    shutil.copy(config_path, out_dir / "config.yaml")

    logger = get_logger(str(out_dir / "train.log"))
    logger.info(f"Experiment : {cfg['experiment']['name']}")
    logger.info(f"Config     : {config_path}")
    logger.info(f"Seed       : {args.seed}")

    device = cfg["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"
    logger.info(f"Device     : {device}")

    # ── Model + processor ─────────────────────────────────────────────────
    model_type = cfg["model"].get("type", "clip")

    if model_type == "clip":
        model_name = cfg["model"]["clip_model"]
        processor  = CLIPProcessor.from_pretrained(model_name)
        model      = CLIPClassifier(cfg)
        max_length = cfg["dataset"].get("max_length", 77)
    elif model_type == "vilt":
        model_name = cfg["model"]["model_name"]
        processor  = ViltProcessor.from_pretrained(model_name)
        model      = ViLTClassifier(cfg)
        max_length = cfg["dataset"].get("max_length", 40)
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")

    logger.info(f"Model      : {model_name}")

    # ── Datasets ──────────────────────────────────────────────────────────
    data_root    = Path(cfg["dataset"]["data_dir"])
    train_dataset = HatefulMemesDataset(data_root, split="train")
    val_dataset   = HatefulMemesDataset(data_root, split="val")
    logger.info(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    collate_fn   = MultimodalCollator(processor, max_length=max_length)
    num_workers  = cfg["training"].get("num_workers", 4)
    batch_size   = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Optimizer + trainer ───────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    trainer   = Trainer(model, train_loader, val_loader, optimizer, device, cfg, logger)
    trainer.train()


if __name__ == "__main__":
    main()
