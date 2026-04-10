"""
Ensemble: CLIP (clip_frozen_concat) + ViLT (vilt_label_smooth).

Averages softmax probabilities from both models.
Evaluates on dev_seen, dev_unseen, test_seen, test_unseen.

Output:
  analysis_outputs/ensemble_results.csv
  analysis_outputs/ensemble_vs_single.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, ViltProcessor

from dataset.hateful_memes import HatefulMemesDataset
from models.clip_classifier import CLIPClassifier
from models.vilt_classifier import ViLTClassifier
from utils.checkpoint import load_checkpoint
from utils.collate import MultimodalCollator
from utils.metrics import compute_metrics

CLIP_CONFIG  = "clip_frozen_concat"
VILT_CONFIG  = "vilt_label_smooth"
SPLITS       = ["dev_seen", "dev_unseen", "test_seen", "test_unseen"]
OUTPUT_DIR   = Path("analysis_outputs")
SUMMARY_CSV  = Path("results_summary.csv")


def load_model_and_processor(config_name: str, device: str):
    config_path = Path("configs") / f"{config_name}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["lr"] = float(cfg["training"]["lr"])

    model_type = cfg["model"].get("type", "clip")
    if model_type == "clip":
        processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])
        model     = CLIPClassifier(cfg)
        max_len   = cfg["dataset"].get("max_length", 77)
    else:
        processor = ViltProcessor.from_pretrained(cfg["model"]["model_name"])
        model     = ViLTClassifier(cfg)
        max_len   = cfg["dataset"].get("max_length", 40)

    ckpt_path = Path(cfg["experiment"]["output_dir"]) / "best.pt"
    load_checkpoint(str(ckpt_path), model, device=device)
    model.to(device).eval()
    return model, processor, max_len, cfg


@torch.no_grad()
def get_probs(model, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    all_probs, all_labels = [], []
    for batch in tqdm(loader, leave=False):
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def build_loader(split: str, processor, max_len: int, batch_size: int = 32):
    ds = HatefulMemesDataset(Path("data"), split=split)
    collate = MultimodalCollator(processor, max_length=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=collate)


def load_single_auc(config_name: str) -> float:
    with open(SUMMARY_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == config_name:
                return float(row["val_auroc"])
    return 0.0


def plot_comparison(results: list[dict], out_path: Path) -> None:
    splits = SPLITS
    x = range(len(splits))
    width = 0.25

    clip_aucs = [r["clip_auc"]     for r in results]
    vilt_aucs = [r["vilt_auc"]     for r in results]
    ens_aucs  = [r["ensemble_auc"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], clip_aucs, width, label=f"CLIP ({CLIP_CONFIG})", color="#457B9D")
    ax.bar([i         for i in x], vilt_aucs, width, label=f"ViLT ({VILT_CONFIG})", color="#E76F51")
    ax.bar([i + width for i in x], ens_aucs,  width, label="Ensemble (avg)",         color="#2A9D8F")

    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Ensemble vs Individual Models — ROC-AUC by Split")
    ax.set_ylim(0.6, 0.85)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading CLIP model...")
    clip_model, clip_proc, clip_maxlen, clip_cfg = load_model_and_processor(CLIP_CONFIG, device)

    print("Loading ViLT model...")
    vilt_model, vilt_proc, vilt_maxlen, vilt_cfg = load_model_and_processor(VILT_CONFIG, device)

    results = []
    print("\n=== Evaluating on all splits ===\n")

    for split in SPLITS:
        print(f"Split: {split}")

        clip_loader = build_loader(split, clip_proc, clip_maxlen)
        vilt_loader = build_loader(split, vilt_proc, vilt_maxlen)

        clip_probs, labels = get_probs(clip_model, clip_loader, device)
        vilt_probs, _      = get_probs(vilt_model, vilt_loader, device)

        # Ensemble: average probabilities
        ens_probs  = (clip_probs + vilt_probs) / 2
        # Convert to logits-like for metrics (just use probs directly)
        ens_logits = torch.log(ens_probs + 1e-8)

        clip_metrics = compute_metrics(torch.log(clip_probs + 1e-8), labels)
        vilt_metrics = compute_metrics(torch.log(vilt_probs + 1e-8), labels)
        ens_metrics  = compute_metrics(ens_logits, labels)

        row = {
            "split":        split,
            "clip_auc":     clip_metrics["auroc"],
            "clip_acc":     clip_metrics["accuracy"],
            "clip_f1":      clip_metrics["f1"],
            "vilt_auc":     vilt_metrics["auroc"],
            "vilt_acc":     vilt_metrics["accuracy"],
            "vilt_f1":      vilt_metrics["f1"],
            "ensemble_auc": ens_metrics["auroc"],
            "ensemble_acc": ens_metrics["accuracy"],
            "ensemble_f1":  ens_metrics["f1"],
        }
        results.append(row)

        print(f"  CLIP      AUC={clip_metrics['auroc']:.4f}  Acc={clip_metrics['accuracy']:.4f}  F1={clip_metrics['f1']:.4f}")
        print(f"  ViLT      AUC={vilt_metrics['auroc']:.4f}  Acc={vilt_metrics['accuracy']:.4f}  F1={vilt_metrics['f1']:.4f}")
        print(f"  Ensemble  AUC={ens_metrics['auroc']:.4f}  Acc={ens_metrics['accuracy']:.4f}  F1={ens_metrics['f1']:.4f}")
        print()

    # Save CSV
    csv_path = OUTPUT_DIR / "ensemble_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved: {csv_path}")

    plot_comparison(results, OUTPUT_DIR / "ensemble_vs_single.png")


if __name__ == "__main__":
    main()
