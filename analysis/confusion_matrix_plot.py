"""
Confusion matrix visualization for the best CLIP and ViLT models.

Reads eval_{split}_multimodal.json saved by evaluate.py.

Output:
  analysis_outputs/fig10_confusion_matrix_clip.png
  analysis_outputs/fig10_confusion_matrix_vilt.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BEST_CLIP = "clip_frozen_concat"
BEST_VILT = "vilt_label_smooth"
SPLIT     = "dev_seen"
OUT_DIR   = Path("analysis_outputs")


def load_confusion_matrix(exp_name: str, split: str) -> np.ndarray | None:
    path = Path("outputs") / exp_name / f"eval_{split}_multimodal.json"
    if not path.exists():
        print(f"  [warning] {path} not found — run evaluate.py first.")
        return None
    with open(path) as f:
        data = json.load(f)
    cm = data.get("confusion_matrix")
    if cm is None:
        return None
    return np.array(cm)


def plot_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Benign (0)", "Hateful (1)"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=12)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")

    # Annotate quadrant labels
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.35, labels[i][j], ha="center", va="center",
                    fontsize=9, color="gray")

    total = cm.sum()
    acc   = (cm[0, 0] + cm[1, 1]) / total if total > 0 else 0
    ax.set_xlabel(
        f"Predicted label  |  Accuracy: {acc:.4f}  |  n={total}",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Confusion matrices on split: {SPLIT}\n")

    for exp_name, tag in [(BEST_CLIP, "clip"), (BEST_VILT, "vilt")]:
        cm = load_confusion_matrix(exp_name, SPLIT)
        if cm is None:
            continue
        model_label = "CLIP (frozen+concat)" if tag == "clip" else "ViLT (label smooth)"
        title = f"Confusion Matrix — {model_label}\n({SPLIT})"
        out_path = OUT_DIR / f"fig10_confusion_matrix_{tag}.png"
        plot_cm(cm, title, out_path)


if __name__ == "__main__":
    main()
