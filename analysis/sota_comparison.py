"""
SOTA comparison table.

Compares our best results against published baselines from:
  - Kiela et al. 2020, "The Hateful Memes Challenge" (original paper)
  - Subsequent challenge entries

Output:
  analysis_outputs/sota_comparison.csv
  analysis_outputs/sota_comparison.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Published baselines ────────────────────────────────────────────────────────
# Source: Kiela et al. (2020), The Hateful Memes Challenge, NeurIPS
# AUC values on test_seen split (Phase 1 leaderboard)
BASELINES = [
    # (name, auc, type, source)
    ("Human",                   0.8265, "human",      "Kiela et al. 2020"),
    ("Text-only (BERT)",        0.6508, "unimodal",   "Kiela et al. 2020"),
    ("Image-only (ResNet-152)", 0.5893, "unimodal",   "Kiela et al. 2020"),
    ("Late Fusion (concat)",    0.6733, "multimodal",  "Kiela et al. 2020"),
    ("MMBT-Grid",               0.7045, "multimodal",  "Kiela et al. 2020"),
    ("VisualBERT (COCO)",       0.7133, "multimodal",  "Kiela et al. 2020"),
    ("ViLBERT (CC)",            0.7061, "multimodal",  "Kiela et al. 2020"),
    ("UNITER",                  0.7390, "multimodal",  "Chen et al. 2020"),
]

# ── Our results (from results_summary.csv) ─────────────────────────────────────
# Val AUC — note: published models use test_seen, ours is val split
OUR_RESULTS_CSV = Path("results_summary.csv")
OUTPUT_DIR = Path("analysis_outputs")


def load_our_results() -> list[tuple]:
    results = []
    with open(OUR_RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append((
                row["experiment"],
                float(row["val_auroc"]),
                "multimodal",
                "ours",
            ))
    return results


def build_comparison_table(our_results: list) -> pd.DataFrame:
    rows = []

    # Published baselines
    for name, auc, model_type, source in BASELINES:
        rows.append({
            "model":  name,
            "auc":    auc,
            "type":   model_type,
            "source": source,
            "ours":   False,
        })

    # Our top models (sorted by AUC, top 5)
    top_ours = sorted(our_results, key=lambda x: -x[1])[:5]
    for name, auc, model_type, source in top_ours:
        rows.append({
            "model":  f"[Ours] {name}",
            "auc":    auc,
            "type":   model_type,
            "source": "this work (val AUC)",
            "ours":   True,
        })

    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    return df


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = []
    for _, row in df.iterrows():
        if row["model"] == "Human":
            colors.append("#888888")
        elif row["ours"]:
            colors.append("#E63946")
        elif row["type"] == "unimodal":
            colors.append("#A8DADC")
        else:
            colors.append("#457B9D")

    bars = ax.barh(df["model"], df["auc"], color=colors, edgecolor="white", height=0.6)

    # Value labels
    for bar, val in zip(bars, df["auc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#E63946", label="Our models (val AUC)"),
        mpatches.Patch(color="#457B9D", label="Published multimodal"),
        mpatches.Patch(color="#A8DADC", label="Published unimodal"),
        mpatches.Patch(color="#888888", label="Human"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.set_xlabel("ROC-AUC", fontsize=11)
    ax.set_title("Hateful Memes: ROC-AUC Comparison with Published Baselines", fontsize=13)
    ax.set_xlim(0.5, 0.92)
    ax.axvline(x=0.5, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    our_results = load_our_results()
    df = build_comparison_table(our_results)

    # Save CSV
    csv_path = OUTPUT_DIR / "sota_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Table saved: {csv_path}")

    # Print table
    print("\n" + "=" * 65)
    print(f"  {'Model':<35} {'AUC':>6}  Source")
    print("-" * 65)
    for _, row in df.iterrows():
        marker = " *" if row["ours"] else ""
        print(f"  {row['model']:<35} {row['auc']:.4f}  {row['source']}{marker}")
    print("=" * 65)
    print("  * val AUC (our models); published baselines use test_seen\n")

    # Plot
    plot_comparison(df, OUTPUT_DIR / "sota_comparison.png")


if __name__ == "__main__":
    main()
