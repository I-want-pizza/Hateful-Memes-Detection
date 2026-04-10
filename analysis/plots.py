"""
Training curve and result visualizations.

Generates figures referenced in the thesis:
  Рис. 5  — LoRA rank vs ROC-AUC (CLIP and ViLT)
  Рис. 6  — Parameter efficiency scatter (trainable params vs Val ROC-AUC)
  Рис. 7  — Train loss by epoch for key experiments
  Рис. 8  — Val ROC-AUC by epoch for key experiments
  Рис. 9  — Fusion method comparison bar chart

Output: analysis_outputs/plots/
"""

from pathlib import Path

import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

RESULTS_CSV = Path("results_summary.csv")
OUTPUTS_DIR = Path("outputs")
OUT_DIR     = Path("analysis_outputs/plots")

# Key experiments shown on learning curves
KEY_EXPERIMENTS = [
    ("clip_frozen_concat",  "CLIP Frozen+Concat (B1)", "#457B9D", "-"),
    ("vilt_full_ft",        "ViLT Full FT (B2)",        "#E76F51", "-"),
    ("clip_lora_r8",        "CLIP LoRA r=8 (C3)",        "#2A9D8F", "--"),
    ("vilt_lora_r8",        "ViLT LoRA r=8 (V3)",        "#E9C46A", "--"),
]

# Fusion experiments (frozen CLIP encoder)
FUSION_EXPERIMENTS = [
    ("clip_frozen_concat",   "Concat"),
    ("clip_frozen_elemwise", "Elemwise"),
    ("clip_frozen_gated",    "Gated"),
    ("clip_frozen_xattn1",   "CrossAttn×1"),
    ("clip_frozen_xattn2",   "CrossAttn×2"),
]

# LoRA rank experiments
LORA_CLIP_EXP = [
    ("clip_lora_r4",  4),
    ("clip_lora_r8",  8),
    ("clip_lora_r16", 16),
    ("clip_lora_r32", 32),
]
LORA_VILT_EXP = [
    ("vilt_lora_r4",  4),
    ("vilt_lora_r8",  8),
    ("vilt_lora_r16", 16),
]


# ─────────────────────────────────────────────────────────────────────────────

def load_results_summary() -> dict[str, dict]:
    """Load results_summary.csv -> {experiment_name: row_dict}."""
    if not RESULTS_CSV.exists():
        return {}
    with open(RESULTS_CSV, newline="") as f:
        return {row["experiment"]: row for row in csv.DictReader(f)}


def load_metrics_csv(exp_name: str) -> list[dict]:
    """Load per-epoch metrics.csv for one experiment."""
    path = OUTPUTS_DIR / exp_name / "metrics.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — LoRA rank vs ROC-AUC
# ─────────────────────────────────────────────────────────────────────────────

def plot_lora_rank(summary: dict[str, dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    def _plot_series(experiments, color, marker, label_prefix):
        ranks, aucs = [], []
        for exp, rank in experiments:
            row = summary.get(exp)
            if row and row.get("val_auroc"):
                ranks.append(rank)
                aucs.append(float(row["val_auroc"]))
        if ranks:
            ax.plot(ranks, aucs, marker=marker, color=color,
                    linewidth=1.8, markersize=7, label=label_prefix)
            for r, a in zip(ranks, aucs):
                ax.annotate(f"{a:.4f}", (r, a),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
        return ranks, aucs

    _plot_series(LORA_CLIP_EXP, "#457B9D", "o", "CLIP LoRA (concat)")
    _plot_series(LORA_VILT_EXP, "#E76F51", "s", "ViLT LoRA")

    ax.set_xlabel("LoRA rank (r)", fontsize=11)
    ax.set_ylabel("Val ROC-AUC", fontsize=11)
    ax.set_title("LoRA Rank vs Validation ROC-AUC", fontsize=12)
    ax.set_xticks([4, 8, 16, 32])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    path = out_dir / "fig5_lora_rank_vs_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Parameter efficiency scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_param_efficiency(summary: dict[str, dict], out_dir: Path) -> None:
    clip_params, clip_aucs, clip_labels = [], [], []
    vilt_params, vilt_aucs, vilt_labels = [], [], []

    for exp_name, row in summary.items():
        if not row.get("val_auroc") or not row.get("trainable_params"):
            continue
        params = float(row["trainable_params"])
        auc    = float(row["val_auroc"])
        label  = exp_name.replace("clip_", "").replace("vilt_", "").replace("_", " ")

        if row["family"] == "CLIP":
            clip_params.append(params)
            clip_aucs.append(auc)
            clip_labels.append(label)
        else:
            vilt_params.append(params)
            vilt_aucs.append(auc)
            vilt_labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))

    if clip_params:
        ax.scatter(clip_params, clip_aucs, color="#457B9D", s=70,
                   zorder=3, label="CLIP", edgecolors="white", linewidths=0.5)
        for p, a, l in zip(clip_params, clip_aucs, clip_labels):
            ax.annotate(l, (p, a), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="#457B9D")

    if vilt_params:
        ax.scatter(vilt_params, vilt_aucs, color="#E76F51", s=70,
                   zorder=3, label="ViLT", marker="s",
                   edgecolors="white", linewidths=0.5)
        for p, a, l in zip(vilt_params, vilt_aucs, vilt_labels):
            ax.annotate(l, (p, a), textcoords="offset points",
                        xytext=(4, 3), fontsize=7, color="#E76F51")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)", fontsize=11)
    ax.set_ylabel("Val ROC-AUC", fontsize=11)
    ax.set_title("Parameter Efficiency: Trainable Params vs Val ROC-AUC", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.35, which="both")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    path = out_dir / "fig6_param_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Train loss by epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_train_loss(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for exp_name, label, color, ls in KEY_EXPERIMENTS:
        rows = load_metrics_csv(exp_name)
        if not rows:
            continue
        epochs = [int(r["epoch"])     for r in rows]
        losses = [float(r["train_loss"]) for r in rows]
        ax.plot(epochs, losses, color=color, linestyle=ls,
                linewidth=1.8, label=label)
        any_data = True

    if not any_data:
        print("  [fig7] No metrics.csv found — skipping.")
        plt.close()
        return

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("Training Loss by Epoch", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    fig.tight_layout()
    path = out_dir / "fig7_train_loss.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Val ROC-AUC by epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_val_auc(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for exp_name, label, color, ls in KEY_EXPERIMENTS:
        rows = load_metrics_csv(exp_name)
        if not rows:
            continue
        epochs = [int(r["epoch"])        for r in rows]
        aucs   = [float(r["val_auroc"])   for r in rows]
        ax.plot(epochs, aucs, color=color, linestyle=ls,
                linewidth=1.8, label=label)
        any_data = True

    if not any_data:
        print("  [fig8] No metrics.csv found — skipping.")
        plt.close()
        return

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val ROC-AUC", fontsize=11)
    ax.set_title("Validation ROC-AUC by Epoch", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.35)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    path = out_dir / "fig8_val_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Fusion method comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_fusion_comparison(summary: dict[str, dict], out_dir: Path) -> None:
    names, aucs, accs = [], [], []
    for exp_name, display_name in FUSION_EXPERIMENTS:
        row = summary.get(exp_name)
        if row and row.get("val_auroc"):
            names.append(display_name)
            aucs.append(float(row["val_auroc"]))
            accs.append(float(row.get("val_accuracy", 0)))

    if not names:
        print("  [fig9] No fusion results — skipping.")
        return

    x     = np.arange(len(names))
    width = 0.35
    colors_auc = "#457B9D"
    colors_acc = "#A8DADC"

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_auc = ax.bar(x - width / 2, aucs, width, label="ROC-AUC",
                      color=colors_auc, edgecolor="white")
    bars_acc = ax.bar(x + width / 2, accs, width, label="Accuracy",
                      color=colors_acc, edgecolor="white")

    ax.bar_label(bars_auc, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(bars_acc, fmt="%.4f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Fusion Method Comparison (frozen CLIP encoder)", fontsize=12)
    ax.set_ylim(0.5, max(max(aucs), max(accs)) + 0.06)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    path = out_dir / "fig9_fusion_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_results_summary()

    if not summary:
        print(f"  WARNING: {RESULTS_CSV} not found — skipping result-based plots.")

    print("[Fig 5]  LoRA rank vs ROC-AUC...")
    plot_lora_rank(summary, OUT_DIR)

    print("[Fig 6]  Parameter efficiency scatter...")
    plot_param_efficiency(summary, OUT_DIR)

    print("[Fig 7]  Train loss by epoch...")
    plot_train_loss(OUT_DIR)

    print("[Fig 8]  Val ROC-AUC by epoch...")
    plot_val_auc(OUT_DIR)

    print("[Fig 9]  Fusion method comparison...")
    plot_fusion_comparison(summary, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
