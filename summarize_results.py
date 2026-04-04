"""
Compile all experiment results into a single summary CSV.

Reads metrics.csv from every outputs/<experiment>/ directory,
extracts best-epoch row, and writes results_summary.csv.
"""
import csv
import json
from pathlib import Path


ALL_EXPERIMENTS = [
    # Baselines
    ("B1", "clip_frozen_concat",   "CLIP", "frozen",      "concat"),
    ("B2", "vilt_full_ft",         "ViLT", "full_ft",     "early"),
    # CLIP encoder strategies
    ("C2", "clip_lora_r4",         "CLIP", "lora_r4",     "concat"),
    ("C3", "clip_lora_r8",         "CLIP", "lora_r8",     "concat"),
    ("C4", "clip_lora_r16",        "CLIP", "lora_r16",    "concat"),
    ("C5", "clip_lora_r32",        "CLIP", "lora_r32",    "concat"),
    ("C6", "clip_unfreeze_1",      "CLIP", "unfreeze_1",  "concat"),
    ("C7", "clip_unfreeze_3",      "CLIP", "unfreeze_3",  "concat"),
    ("C8", "clip_unfreeze_6",      "CLIP", "unfreeze_6",  "concat"),
    # CLIP fusion strategies (all frozen)
    ("F2", "clip_frozen_elemwise", "CLIP", "frozen",      "elemwise"),
    ("F3", "clip_frozen_gated",    "CLIP", "frozen",      "gated"),
    ("F4", "clip_frozen_xattn1",   "CLIP", "frozen",      "crossattn1"),
    ("F5", "clip_frozen_xattn2",   "CLIP", "frozen",      "crossattn2"),
    # Best combination
    ("CF", "clip_lora_r8_xattn1",  "CLIP", "lora_r8",    "crossattn1"),
    # ViLT strategies
    ("V2", "vilt_lora_r4",         "ViLT", "lora_r4",    "early"),
    ("V3", "vilt_lora_r8",         "ViLT", "lora_r8",    "early"),
    ("V4", "vilt_lora_r16",        "ViLT", "lora_r16",   "early"),
    ("V5", "vilt_unfreeze_3",      "ViLT", "unfreeze_3", "early"),
    ("V6", "vilt_unfreeze_6",      "ViLT", "unfreeze_6", "early"),
    ("V7", "vilt_label_smooth",    "ViLT", "full_ft+ls", "early"),
    ("V8", "vilt_dropout_smooth",  "ViLT", "full_ft+reg","early"),
]


def read_best_metrics(exp_name: str) -> dict:
    """Read metrics.csv and return the row with highest val_auroc."""
    csv_path = Path("outputs") / exp_name / "metrics.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return max(rows, key=lambda r: float(r.get("val_auroc", 0)))


def read_trainable_params(exp_name: str) -> str:
    """Read trainable param count from sysinfo.txt."""
    path = Path("outputs") / exp_name / "sysinfo.txt"
    if not path.exists():
        return ""
    with open(path) as f:
        for line in f:
            if "Trainable params" in line:
                return line.split(":")[1].strip().replace(",", "")
    return ""


def main() -> None:
    rows = []
    for exp_id, name, family, encoder, fusion in ALL_EXPERIMENTS:
        m = read_best_metrics(name)
        trainable = read_trainable_params(name)
        rows.append({
            "id":              exp_id,
            "experiment":      name,
            "family":          family,
            "encoder":         encoder,
            "fusion":          fusion,
            "trainable_params": trainable,
            "best_epoch":      m.get("epoch", ""),
            "val_accuracy":    m.get("val_accuracy", ""),
            "val_auroc":       m.get("val_auroc", ""),
            "val_f1":          m.get("val_f1", ""),
            "val_precision":   m.get("val_precision", ""),
            "val_recall":      m.get("val_recall", ""),
        })

    out_path = Path("results_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results summary saved to {out_path}")
    print(f"\n{'ID':<4} {'Experiment':<28} {'AUC':>6} {'Acc':>6} {'F1':>6}")
    print("-" * 55)
    for r in rows:
        auc = r["val_auroc"]
        acc = r["val_accuracy"]
        f1  = r["val_f1"]
        if auc:
            print(f"{r['id']:<4} {r['experiment']:<28} {float(auc):>6.4f} {float(acc):>6.4f} {float(f1):>6.4f}")
        else:
            print(f"{r['id']:<4} {r['experiment']:<28} {'—':>6} {'—':>6} {'—':>6}")


if __name__ == "__main__":
    main()
