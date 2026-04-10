"""
Error analysis: find and visualize hard examples.

Categories:
  both_wrong_fp  — both models predict hateful, true label: benign
  both_wrong_fn  — both models predict benign,  true label: hateful
  clip_only      — only CLIP correct
  vilt_only      — only ViLT correct

Output:
  analysis_outputs/error_analysis/both_wrong_fp/  (images + info)
  analysis_outputs/error_analysis/both_wrong_fn/
  analysis_outputs/error_analysis/clip_only/
  analysis_outputs/error_analysis/vilt_only/
  analysis_outputs/error_analysis/summary.csv
  analysis_outputs/error_analysis_grid.png
"""

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, ViltProcessor

from dataset.hateful_memes import HatefulMemesDataset
from models.clip_classifier import CLIPClassifier
from models.vilt_classifier import ViLTClassifier
from utils.checkpoint import load_checkpoint
from utils.collate import MultimodalCollator

CLIP_CONFIG = "clip_frozen_concat"
VILT_CONFIG = "vilt_label_smooth"
SPLIT       = "dev_seen"
TOP_K       = 20           # examples to save per category
OUTPUT_DIR  = Path("analysis_outputs/error_analysis")
DATA_DIR    = Path("data")


def load_model(config_name: str, device: str):
    with open(Path("configs") / f"{config_name}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["lr"] = float(cfg["training"]["lr"])

    if cfg["model"].get("type", "clip") == "clip":
        processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])
        model     = CLIPClassifier(cfg)
        max_len   = cfg["dataset"].get("max_length", 77)
    else:
        processor = ViltProcessor.from_pretrained(cfg["model"]["model_name"])
        model     = ViLTClassifier(cfg)
        max_len   = cfg["dataset"].get("max_length", 40)

    load_checkpoint(str(Path(cfg["experiment"]["output_dir"]) / "best.pt"), model, device=device)
    model.to(device).eval()
    return model, processor, max_len


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    all_probs, all_labels = [], []
    for batch in tqdm(loader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
        all_probs.append(F.softmax(logits, dim=-1).cpu())
        all_labels.append(batch["labels"].cpu())
    return torch.cat(all_probs), torch.cat(all_labels)


def confidence(probs: torch.Tensor) -> torch.Tensor:
    """Confidence = probability assigned to predicted class."""
    preds = probs.argmax(dim=1)
    return probs[range(len(probs)), preds]


def save_example_image(idx: int, dataset_items: list, category: str,
                        clip_pred: int, vilt_pred: int,
                        clip_conf: float, vilt_conf: float,
                        out_dir: Path) -> dict:
    item    = dataset_items[idx]
    img_path = DATA_DIR / item["img"]
    label   = item["label"]
    text    = item["text"]

    # Copy image
    dest_img = out_dir / f"{idx:04d}_{Path(img_path).stem}.png"
    shutil.copy(img_path, dest_img)

    # Save metadata
    meta = {
        "index":     idx,
        "id":        item.get("id", idx),
        "text":      text,
        "true_label": label,
        "clip_pred":  clip_pred,
        "vilt_pred":  vilt_pred,
        "clip_conf":  round(clip_conf, 4),
        "vilt_conf":  round(vilt_conf, 4),
        "category":   category,
        "image_file": dest_img.name,
    }
    with open(out_dir / f"{idx:04d}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def make_grid(metas: list[dict], category: str, out_path: Path, n_cols: int = 5) -> None:
    metas = metas[:20]
    n_rows = (len(metas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 4))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else list(axes)

    label_map = {0: "benign", 1: "hateful"}

    for i, meta in enumerate(metas):
        ax = axes[i]
        img_path = DATA_DIR / meta["img_rel"]
        try:
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#cccccc")

        true_str = label_map[meta["true_label"]]
        clip_str = label_map[meta["clip_pred"]]
        vilt_str = label_map[meta["vilt_pred"]]
        text     = meta["text"][:60] + "…" if len(meta["text"]) > 60 else meta["text"]

        ax.set_title(
            f'"{text}"\n'
            f"True: {true_str}\n"
            f"CLIP: {clip_str} ({meta['clip_conf']:.2f})\n"
            f"ViLT: {vilt_str} ({meta['vilt_conf']:.2f})",
            fontsize=7, loc="left",
        )
        ax.axis("off")

    for j in range(len(metas), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Error Analysis — {category}", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Grid saved: {out_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CLIP...")
    clip_model, clip_proc, clip_maxlen = load_model(CLIP_CONFIG, device)
    print("Loading ViLT...")
    vilt_model, vilt_proc, vilt_maxlen = load_model(VILT_CONFIG, device)

    # Load dataset (keep raw items for image paths / text)
    ds = HatefulMemesDataset(DATA_DIR, split=SPLIT)
    items = ds.data  # list of dicts: {id, img, text, label}

    clip_loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                             collate_fn=MultimodalCollator(clip_proc, clip_maxlen))
    vilt_loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                             collate_fn=MultimodalCollator(vilt_proc, vilt_maxlen))

    print(f"\nRunning inference on {SPLIT} ({len(ds)} samples)...")
    clip_probs, labels = collect_predictions(clip_model, clip_loader, device)
    vilt_probs, _      = collect_predictions(vilt_model, vilt_loader, device)

    clip_preds = clip_probs.argmax(dim=1)
    vilt_preds = vilt_probs.argmax(dim=1)
    clip_confs = confidence(clip_probs)
    vilt_confs = confidence(vilt_probs)

    # Categorize
    categories = {
        "both_wrong_fp": [],  # true=0, both predict 1
        "both_wrong_fn": [],  # true=1, both predict 0
        "clip_only":     [],  # clip correct, vilt wrong
        "vilt_only":     [],  # vilt correct, clip wrong
    }

    for i in range(len(labels)):
        true  = labels[i].item()
        cp    = clip_preds[i].item()
        vp    = vilt_preds[i].item()
        cc    = clip_confs[i].item()
        vc    = vilt_confs[i].item()

        if cp == true and vp == true:
            continue  # both correct
        elif cp != true and vp != true:
            key = "both_wrong_fp" if cp == 1 else "both_wrong_fn"
            categories[key].append((i, true, cp, vp, cc, vc))
        elif cp == true and vp != true:
            categories["clip_only"].append((i, true, cp, vp, cc, vc))
        elif vp == true and cp != true:
            categories["vilt_only"].append((i, true, cp, vp, cc, vc))

    # Print stats
    print(f"\n=== Error Analysis on {SPLIT} (n={len(labels)}) ===")
    for cat, errs in categories.items():
        print(f"  {cat:<20}: {len(errs)}")

    # Save examples
    all_metas = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cat, errs in categories.items():
        cat_dir = OUTPUT_DIR / cat
        cat_dir.mkdir(exist_ok=True)

        # Sort by combined confidence (most confident wrong = most interesting)
        errs_sorted = sorted(errs, key=lambda x: -(x[4] + x[5]))

        grid_metas = []
        for entry in errs_sorted[:TOP_K]:
            idx, true, cp, vp, cc, vc = entry
            meta = save_example_image(idx, items, cat, cp, vp, cc, vc, cat_dir)
            meta["img_rel"] = items[idx]["img"]
            all_metas.append(meta)
            grid_metas.append(meta)

        if grid_metas:
            make_grid(grid_metas, cat,
                      Path("analysis_outputs") / f"error_{cat}.png")

    # Save summary CSV
    summary_path = OUTPUT_DIR / "summary.csv"
    if all_metas:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_metas[0].keys())
            writer.writeheader()
            writer.writerows(all_metas)
        print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
