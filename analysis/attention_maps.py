"""
CLIP attention map visualization.

Extracts attention weights from the last layer of CLIP's vision encoder
and overlays them on the original meme image.

For ViT-B/32: 224×224 image → 7×7 = 49 patches + 1 [CLS] token.
We take [CLS] → patch attention, average over heads, reshape to 7×7,
upsample to 224×224 and blend with the original image.

Selects examples from each outcome category:
  TP (true hateful, predicted hateful)
  FP (true benign,  predicted hateful)
  FN (true hateful, predicted benign)
  TN (true benign,  predicted benign)

Output:
  analysis_outputs/attention_maps/<category>/<idx>_attn.png
  analysis_outputs/attention_maps_grid.png
"""

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from dataset.hateful_memes import HatefulMemesDataset
from models.clip_classifier import CLIPClassifier
from utils.checkpoint import load_checkpoint
from utils.collate import MultimodalCollator

CLIP_CONFIG   = "clip_frozen_concat"
SPLIT         = "dev_seen"
N_PER_CAT     = 5          # examples per category (TP, FP, FN, TN)
OUTPUT_DIR    = Path("analysis_outputs/attention_maps")
DATA_DIR      = Path("data")
IMG_SIZE      = 224        # CLIP ViT-B/32 input size
PATCH_SIZE    = 32         # ViT-B/32 patch size
GRID_H        = IMG_SIZE // PATCH_SIZE   # 7
GRID_W        = IMG_SIZE // PATCH_SIZE   # 7


def load_model(device: str):
    with open(Path("configs") / f"{CLIP_CONFIG}.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["training"]["lr"] = float(cfg["training"]["lr"])

    processor = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])
    model     = CLIPClassifier(cfg)
    load_checkpoint(str(Path(cfg["experiment"]["output_dir"]) / "best.pt"), model, device=device)
    model.to(device).eval()
    max_len = cfg["dataset"].get("max_length", 77)
    return model, processor, max_len


@torch.no_grad()
def get_attention_map(model: CLIPClassifier,
                      pixel_values: torch.Tensor) -> np.ndarray:
    """
    Returns attention map (7×7 numpy array, values in [0,1])
    from the last layer of CLIP's vision encoder.
    """
    # Run vision encoder with attention outputs
    # clip is CLIPModel, so access vision_model directly
    vision_outputs = model.clip.vision_model(
        pixel_values=pixel_values,
        output_attentions=True,
        return_dict=True,
    )
    # attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    last_attn = vision_outputs.attentions[-1]   # (1, 12, 50, 50)

    # [CLS] token (index 0) attention to all 49 image patches
    cls_attn = last_attn[0, :, 0, 1:]           # (12, 49) — heads × patches
    cls_attn = cls_attn.mean(0)                  # (49,)    — average over heads
    cls_attn = cls_attn.reshape(GRID_H, GRID_W)  # (7, 7)
    cls_attn = cls_attn.cpu().float().numpy()

    # Normalize to [0, 1]
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
    return cls_attn


def overlay_attention(original_img: Image.Image,
                      attn_map: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """Blend attention heatmap with original image."""
    img_np = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))

    # Upsample attention map to image size
    attn_up = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR
        )
    ) / 255.0

    heatmap = cm.jet(attn_up)[:, :, :3]  # (224, 224, 3) RGB
    blended = (1 - alpha) * img_np / 255.0 + alpha * heatmap
    return np.clip(blended, 0, 1)


def save_attention_image(original_img: Image.Image,
                         attn_map: np.ndarray,
                         text: str,
                         true_label: int,
                         pred_label: int,
                         conf: float,
                         out_path: Path) -> None:
    blended = overlay_attention(original_img, attn_map)
    label_map = {0: "benign", 1: "hateful"}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_img.resize((IMG_SIZE, IMG_SIZE)))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title(
        f"Attention\nPred: {label_map[pred_label]} ({conf:.2f})",
        fontsize=10,
    )
    axes[1].axis("off")

    caption = text[:80] + "…" if len(text) > 80 else text
    fig.suptitle(
        f'"{caption}"\nTrue: {label_map[true_label]}',
        fontsize=9, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def make_summary_grid(examples: list[dict], out_path: Path) -> None:
    """4-column grid: one column per category."""
    categories = ["TP", "TN", "FP", "FN"]
    label_map  = {0: "benign", 1: "hateful"}
    n_rows = N_PER_CAT

    fig, axes = plt.subplots(n_rows, len(categories),
                              figsize=(len(categories) * 3.5, n_rows * 4.5))

    for col, cat in enumerate(categories):
        cat_examples = [e for e in examples if e["category"] == cat][:n_rows]
        for row, ex in enumerate(cat_examples):
            ax = axes[row][col]
            blended = overlay_attention(
                Image.open(DATA_DIR / ex["img_rel"]).convert("RGB"),
                ex["attn_map"],
            )
            ax.imshow(blended)
            text = ex["text"][:40] + "…" if len(ex["text"]) > 40 else ex["text"]
            ax.set_title(
                f'"{text}"\n'
                f"True: {label_map[ex['true_label']]}\n"
                f"Pred: {label_map[ex['pred']]} ({ex['conf']:.2f})",
                fontsize=7,
            )
            ax.axis("off")

        # Column header
        axes[0][col].set_title(
            f"{'✓' if cat in ('TP','TN') else '✗'} {cat}\n" +
            axes[0][col].get_title(),
            fontsize=8, fontweight="bold",
        )

        for row in range(len(cat_examples), n_rows):
            axes[row][col].axis("off")

    fig.suptitle("CLIP Attention Maps — by Prediction Outcome", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Summary grid saved: {out_path}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading CLIP model...")
    model, processor, max_len = load_model(device)

    ds = HatefulMemesDataset(DATA_DIR, split=SPLIT)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=MultimodalCollator(processor, max_length=max_len),
    )
    items = ds.data

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cat in ("TP", "TN", "FP", "FN"):
        (OUTPUT_DIR / cat).mkdir(exist_ok=True)

    # Counts per category
    counts  = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    saved   = {"TP": [], "TN": [], "FP": [], "FN": []}
    all_ex  = []

    print(f"\nExtracting attention maps on {SPLIT}...")

    for i, batch in enumerate(tqdm(loader)):
        # Check if we have enough examples in all categories
        if all(v >= N_PER_CAT for v in counts.values()):
            break

        pixel_values = batch["pixel_values"].to(device)
        input_ids    = batch["input_ids"].to(device)
        attn_mask    = batch["attention_mask"].to(device)
        label        = batch["labels"][0].item()

        # Get prediction
        with torch.no_grad():
            logits = model(pixel_values, input_ids, attn_mask)
            probs  = F.softmax(logits, dim=-1)[0]
            pred   = probs.argmax().item()
            conf   = probs[pred].item()

        # Determine category
        if label == 1 and pred == 1:
            cat = "TP"
        elif label == 0 and pred == 0:
            cat = "TN"
        elif label == 0 and pred == 1:
            cat = "FP"
        else:
            cat = "FN"

        if counts[cat] >= N_PER_CAT:
            continue

        # Extract attention map
        attn_map = get_attention_map(model, pixel_values)

        item = items[i]
        original_img = Image.open(DATA_DIR / item["img"]).convert("RGB")

        out_path = OUTPUT_DIR / cat / f"{i:04d}_attn.png"
        save_attention_image(original_img, attn_map, item["text"],
                             label, pred, conf, out_path)

        all_ex.append({
            "category":   cat,
            "img_rel":    item["img"],
            "text":       item["text"],
            "true_label": label,
            "pred":       pred,
            "conf":       conf,
            "attn_map":   attn_map,
        })

        counts[cat] += 1
        saved[cat].append(i)

    print("\n=== Collected examples ===")
    for cat, cnt in counts.items():
        print(f"  {cat}: {cnt}")

    make_summary_grid(all_ex, Path("analysis_outputs") / "attention_maps_grid.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
