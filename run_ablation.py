"""
Run full modality ablation study for the best CLIP and best ViLT checkpoints.

Determines "best" by reading metrics.csv from each experiment output dir
and finding the highest val_auroc.

Outputs: ablation_results.csv
"""
import csv
import subprocess
import sys
from pathlib import Path


CLIP_EXPERIMENTS = [
    "clip_frozen_concat",
    "clip_lora_r4",
    "clip_lora_r8",
    "clip_lora_r16",
    "clip_lora_r32",
    "clip_unfreeze_1",
    "clip_unfreeze_3",
    "clip_unfreeze_6",
    "clip_frozen_elemwise",
    "clip_frozen_gated",
    "clip_frozen_xattn1",
    "clip_frozen_xattn2",
    "clip_lora_r8_xattn1",
]

VILT_EXPERIMENTS = [
    "vilt_full_ft",
    "vilt_lora_r4",
    "vilt_lora_r8",
    "vilt_lora_r16",
    "vilt_unfreeze_3",
    "vilt_unfreeze_6",
    "vilt_label_smooth",
    "vilt_dropout_smooth",
]

SPLITS    = ["dev_seen", "dev_unseen", "test_seen", "test_unseen"]
ABLATIONS = ["multimodal", "text_only", "image_only"]


def find_best_experiment(experiments: list) -> str:
    """Return experiment name with highest val_auroc from its metrics.csv."""
    best_name  = None
    best_auroc = -1.0
    for name in experiments:
        csv_path = Path("outputs") / name / "metrics.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        auroc = max(float(r["val_auroc"]) for r in rows)
        if auroc > best_auroc:
            best_auroc = auroc
            best_name  = name
    return best_name


def run_evaluate(config: str, split: str, ablation: str) -> dict:
    cmd = [
        sys.executable, "evaluate.py",
        "--config",   config,
        "--split",    split,
        "--ablation", ablation,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {config} {split} {ablation}\n{result.stderr}")
        return {}

    # Parse metrics from JSON saved by evaluate.py
    import json
    json_path = Path("outputs") / config / f"eval_{split}_{ablation}.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


def main() -> None:
    best_clip = find_best_experiment(CLIP_EXPERIMENTS)
    best_vilt = find_best_experiment(VILT_EXPERIMENTS)

    print(f"Best CLIP: {best_clip}")
    print(f"Best ViLT: {best_vilt}")

    rows = []
    for model_name, config in [("CLIP", best_clip), ("ViLT", best_vilt)]:
        if config is None:
            print(f"No trained {model_name} checkpoint found — skipping.")
            continue
        for split in SPLITS:
            for ablation in ABLATIONS:
                print(f"  {config} | {split} | {ablation}")
                result = run_evaluate(config, split, ablation)
                if result:
                    rows.append({
                        "model_family": model_name,
                        "config":       config,
                        "split":        split,
                        "ablation":     ablation,
                        "accuracy":     result.get("accuracy", ""),
                        "auroc":        result.get("auroc", ""),
                        "f1":           result.get("f1", ""),
                        "precision":    result.get("precision", ""),
                        "recall":       result.get("recall", ""),
                    })

    out_path = Path("ablation_results.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nAblation results saved to {out_path}")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
