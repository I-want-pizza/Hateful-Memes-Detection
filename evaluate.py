"""
Evaluate a trained checkpoint on any split.

Usage:
    python evaluate.py --config clip_frozen_concat --split val
    python evaluate.py --config clip_lora_r8 --split dev_seen --ablation text_only
    python evaluate.py --config vilt_lora_r8 --split test
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, ViltProcessor

from dataset.hateful_memes import HatefulMemesDataset
from evaluators.evaluator import Evaluator
from models.clip_classifier import CLIPClassifier
from models.vilt_classifier import ViLTClassifier
from utils.checkpoint import load_checkpoint
from utils.collate import MultimodalCollator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, required=True)
    parser.add_argument("--split",    type=str, default="val",
                        choices=["val", "dev_seen", "dev_unseen",
                                 "test", "test_seen", "test_unseen"])
    parser.add_argument("--ablation", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: outputs/<config>/best.pt)")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    config_path = Path("configs") / f"{args.config}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["training"]["lr"] = float(cfg["training"]["lr"])

    out_dir = Path(cfg["experiment"]["output_dir"])
    ckpt_path = args.checkpoint or str(out_dir / "best.pt")

    device = cfg["training"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # ── Model ─────────────────────────────────────────────────────────────
    model_type = cfg["model"].get("type", "clip")
    if model_type == "clip":
        model_name = cfg["model"]["clip_model"]
        processor  = CLIPProcessor.from_pretrained(model_name)
        model      = CLIPClassifier(cfg)
        max_length = cfg["dataset"].get("max_length", 77)
    else:
        model_name = cfg["model"]["model_name"]
        processor  = ViltProcessor.from_pretrained(model_name)
        model      = ViLTClassifier(cfg)
        max_length = cfg["dataset"].get("max_length", 40)

    load_checkpoint(ckpt_path, model, device=device)

    # ── Dataset ───────────────────────────────────────────────────────────
    data_root = Path(cfg["dataset"]["data_dir"])
    dataset   = HatefulMemesDataset(data_root, split=args.split)

    collate_fn  = MultimodalCollator(processor, max_length=max_length)
    data_loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=0, collate_fn=collate_fn,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    evaluator = Evaluator(model, device=device)
    metrics   = evaluator.evaluate(
        data_loader,
        desc=f"{args.config}",
        ablation=args.ablation,
    )

    # ── Print & save ──────────────────────────────────────────────────────
    result = {
        "config":    args.config,
        "split":     args.split,
        "ablation":  args.ablation,
        "checkpoint": ckpt_path,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": metrics.get("confusion_matrix"),
    }

    print("\n" + "=" * 50)
    print(f"  Config   : {args.config}")
    print(f"  Split    : {args.split}")
    print(f"  Ablation : {args.ablation}")
    print("-" * 50)
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k.upper():<12}: {v:.4f}")
    print(f"  Confusion matrix:")
    cm = metrics.get("confusion_matrix", [])
    if cm:
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print("=" * 50 + "\n")

    # Save result JSON next to checkpoint
    save_name = f"eval_{args.split}_{args.ablation}.json"
    save_path = out_dir / save_name
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
