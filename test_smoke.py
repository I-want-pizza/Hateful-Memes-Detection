"""
Smoke test — быстрая проверка всего кода перед запуском на remote.

Что проверяется:
  1. Все импорты работают
  2. Датасет загружается (реальные данные)
  3. CLIP (frozen + concat)     — forward pass + 1 batch обучения
  4. CLIP (LoRA r=4)            — forward pass
  5. CLIP (unfreeze_1)          — forward pass
  6. CLIP (frozen + crossattn1) — forward pass
  7. CLIP (frozen + elemwise)   — forward pass
  8. CLIP (frozen + gated)      — forward pass
  9. ViLT (full_ft)             — forward pass + 1 batch обучения
 10. ViLT (LoRA r=4)            — forward pass
 11. Evaluator (все 3 ablation-режима)
 12. Метрики (compute_metrics, confusion matrix)
 13. summarize_results.py import

Время: ~2-5 минут на CPU, ~1 минута на GPU.

Usage:
    python test_smoke.py --data-dir D:/path/to/hateful_memes_dataset
    python test_smoke.py --data-dir /mnt/d/path/to/dataset   # WSL
    python test_smoke.py --data-dir data                      # если data/ рядом с кодом
"""

import argparse
import os
import sys
import time
import traceback

# ── Отключить wandb ДО любых импортов trainer ────────────────────────────────
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"

import torch
import yaml

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 2          # минимальный batch для теста
N_SAMPLES = 8       # сколько семплов грузить из датасета


# ── Цветной вывод ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")
def info(msg):  print(f"  {YELLOW}→{RESET} {msg}")


# ── Счётчик ───────────────────────────────────────────────────────────────────
passed = 0
failed = 0

def run_test(name: str, fn):
    global passed, failed
    print(f"\n[{name}]")
    t0 = time.time()
    try:
        fn()
        ok(f"passed in {time.time()-t0:.1f}s")
        passed += 1
    except Exception as e:
        fail(f"FAILED: {e}")
        traceback.print_exc()
        failed += 1


# ── Вспомогательные фабрики конфигов ──────────────────────────────────────────
def clip_cfg(encoder_strategy="frozen", fusion="concat",
             lora_r=4, data_dir="data"):
    cfg = {
        "experiment": {"name": f"smoke_clip_{encoder_strategy}_{fusion}",
                       "output_dir": "outputs/_smoke_test/"},
        "dataset":    {"data_dir": data_dir, "max_length": 77},
        "model": {
            "type": "clip",
            "clip_model": "openai/clip-vit-base-patch32",
            "num_classes": 2,
            "dropout": 0.2,
            "encoder_strategy": encoder_strategy,
            "fusion": fusion,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
            "lora_dropout": 0.1,
        },
        "training": {
            "epochs": 1, "batch_size": BATCH,
            "lr": 1e-4, "weight_decay": 1e-4,
            "fp16": False, "grad_accumulation_steps": 1,
            "early_stopping_patience": 99,
            "label_smoothing": 0.0,
            "num_workers": 0, "device": DEVICE,
        },
        "logging": {"wandb_project": "smoke-test"},
    }
    return cfg


def vilt_cfg(encoder_strategy="full_ft", lora_r=4, data_dir="data"):
    cfg = {
        "experiment": {"name": f"smoke_vilt_{encoder_strategy}",
                       "output_dir": "outputs/_smoke_test/"},
        "dataset":    {"data_dir": data_dir, "max_length": 40, "image_size": 384},
        "model": {
            "type": "vilt",
            "model_name": "dandelin/vilt-b32-mlm",
            "num_classes": 2,
            "hidden_dim": 768,
            "dropout": 0.1,
            "encoder_strategy": encoder_strategy,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
            "lora_dropout": 0.1,
            "gradient_checkpointing": False,
        },
        "training": {
            "epochs": 1, "batch_size": BATCH,
            "lr": 2e-5, "weight_decay": 0.05,
            "fp16": False, "grad_accumulation_steps": 1,
            "early_stopping_patience": 99,
            "label_smoothing": 0.0,
            "num_workers": 0, "device": DEVICE,
        },
        "logging": {"wandb_project": "smoke-test"},
    }
    return cfg


# ── Построение загрузчика для N семплов ──────────────────────────────────────
def make_loader(cfg, split="train"):
    from pathlib import Path
    from torch.utils.data import DataLoader, Subset
    from transformers import CLIPProcessor, ViltProcessor
    from dataset.hateful_memes import HatefulMemesDataset
    from utils.collate import MultimodalCollator

    model_type = cfg["model"]["type"]
    if model_type == "clip":
        proc       = CLIPProcessor.from_pretrained(cfg["model"]["clip_model"])
        max_length = cfg["dataset"].get("max_length", 77)
    else:
        proc       = ViltProcessor.from_pretrained(cfg["model"]["model_name"])
        max_length = cfg["dataset"].get("max_length", 40)

    ds     = HatefulMemesDataset(Path(cfg["dataset"]["data_dir"]), split=split)
    subset = Subset(ds, list(range(min(N_SAMPLES, len(ds)))))
    loader = DataLoader(subset, batch_size=BATCH, shuffle=False,
                        num_workers=0, collate_fn=MultimodalCollator(proc, max_length))
    return loader


def make_model(cfg):
    from models.clip_classifier import CLIPClassifier
    from models.vilt_classifier import ViLTClassifier
    if cfg["model"]["type"] == "clip":
        return CLIPClassifier(cfg).to(DEVICE)
    else:
        return ViLTClassifier(cfg).to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# ТЕСТЫ
# ═══════════════════════════════════════════════════════════════════════════════

def test_imports():
    from dataset.hateful_memes import HatefulMemesDataset
    from models.clip_classifier import CLIPClassifier
    from models.vilt_classifier import ViLTClassifier
    from trainers.trainer import Trainer
    from evaluators.evaluator import Evaluator
    from utils.metrics import compute_metrics, compute_confusion_matrix
    from utils.collate import MultimodalCollator
    from utils.checkpoint import save_checkpoint, load_checkpoint
    from utils.logger import get_logger
    info("all modules imported successfully")


def test_dataset(data_dir):
    from pathlib import Path
    from dataset.hateful_memes import HatefulMemesDataset, SPLIT_FILES

    for split in list(SPLIT_FILES.keys()):
        try:
            ds = HatefulMemesDataset(Path(data_dir), split=split)
            img, label, text = ds[0]
            assert img is not None and isinstance(label, int) and isinstance(text, str)
            info(f"split={split:<12}  len={len(ds):>5}  label={label}  text='{text[:30]}...'")
        except FileNotFoundError as e:
            info(f"split={split:<12}  skipped ({e})")


def test_clip_forward(data_dir, encoder_strategy="frozen", fusion="concat", **kw):
    cfg    = clip_cfg(encoder_strategy, fusion, data_dir=data_dir, **kw)
    model  = make_model(cfg)
    loader = make_loader(cfg)
    batch  = next(iter(loader))
    batch  = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (min(BATCH, N_SAMPLES), 2)
    trainable = model.count_trainable_params()
    info(f"encoder={encoder_strategy}  fusion={fusion}  trainable={trainable:,}  logits={logits.shape}")


def test_vilt_forward(data_dir, encoder_strategy="full_ft", **kw):
    cfg    = vilt_cfg(encoder_strategy, data_dir=data_dir, **kw)
    model  = make_model(cfg)
    loader = make_loader(cfg)
    batch  = next(iter(loader))
    batch  = {k: v.to(DEVICE) for k, v in batch.items()}

    with torch.no_grad():
        logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (min(BATCH, N_SAMPLES), 2)
    trainable = model.count_trainable_params()
    info(f"encoder={encoder_strategy}  trainable={trainable:,}  logits={logits.shape}")


def test_training_step_clip(data_dir):
    """1 батч обучения CLIP — проверяем backward + шаг оптимизатора."""
    cfg    = clip_cfg(data_dir=data_dir)
    model  = make_model(cfg)
    loader = make_loader(cfg)
    opt    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
    loss   = criterion(logits, batch["labels"])
    loss.backward()
    opt.step()
    info(f"loss={loss.item():.4f}  OK")


def test_training_step_vilt(data_dir):
    """1 батч обучения ViLT."""
    cfg    = vilt_cfg(data_dir=data_dir)
    model  = make_model(cfg)
    loader = make_loader(cfg)
    opt    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5
    )
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    batch = next(iter(loader))
    batch = {k: v.to(DEVICE) for k, v in batch.items()}
    logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
    loss   = criterion(logits, batch["labels"])
    loss.backward()
    opt.step()
    info(f"loss={loss.item():.4f}  OK")


def test_ablation_modes(data_dir):
    """Проверяем все 3 ablation-режима (multimodal, text_only, image_only)."""
    from evaluators.evaluator import Evaluator

    cfg    = clip_cfg(data_dir=data_dir)
    model  = make_model(cfg)
    loader = make_loader(cfg)
    ev     = Evaluator(model, device=DEVICE)

    for mode in ("multimodal", "text_only", "image_only"):
        metrics = ev.evaluate(loader, desc="smoke", ablation=mode)
        info(f"mode={mode:<12}  acc={metrics['accuracy']:.4f}  auc={metrics['auroc']:.4f}")

    # Также ViLT ablation
    cfg2   = vilt_cfg(data_dir=data_dir)
    model2 = make_model(cfg2)
    loader2 = make_loader(cfg2)
    ev2    = Evaluator(model2, device=DEVICE)
    for mode in ("multimodal", "text_only", "image_only"):
        metrics = ev2.evaluate(loader2, desc="smoke_vilt", ablation=mode)
        info(f"ViLT mode={mode:<12}  acc={metrics['accuracy']:.4f}")


def test_metrics():
    """compute_metrics на синтетических данных."""
    from utils.metrics import compute_metrics, compute_confusion_matrix

    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = torch.tensor([1, 0, 1, 0])
    m  = compute_metrics(logits, labels)
    cm = compute_confusion_matrix(logits, labels)
    info(f"acc={m['accuracy']:.4f}  auc={m['auroc']:.4f}  f1={m['f1']:.4f}")
    info(f"confusion matrix:\n      TN={cm[0,0]}  FP={cm[0,1]}\n      FN={cm[1,0]}  TP={cm[1,1]}")
    assert m["accuracy"] == 1.0, "Expected perfect accuracy on simple example"


def test_checkpoint(data_dir, tmp_path="outputs/_smoke_test/ckpt_test.pt"):
    """Сохранение и загрузка чекпоинта."""
    import os
    from utils.checkpoint import save_checkpoint, load_checkpoint

    cfg   = clip_cfg(data_dir=data_dir)
    model = make_model(cfg)
    opt   = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    save_checkpoint(model, opt, epoch=1, metrics={"auroc": 0.75}, cfg=cfg, path=tmp_path)
    assert os.path.exists(tmp_path)

    # Load into new model instance
    model2 = make_model(cfg)
    ckpt   = load_checkpoint(tmp_path, model2, device=DEVICE)
    assert ckpt["epoch"] == 1
    assert abs(ckpt["metrics"]["auroc"] - 0.75) < 1e-6
    info(f"checkpoint saved and loaded OK  ({os.path.getsize(tmp_path)/1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to Hateful Memes dataset directory (containing img/, train.jsonl, etc.)"
    )
    parser.add_argument(
        "--skip-vilt", action="store_true",
        help="Skip ViLT tests (faster if you only want to test CLIP)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Hateful Memes — Smoke Test")
    print(f"  Device   : {DEVICE}")
    print(f"  data-dir : {args.data_dir}")
    print(f"  skip-vilt: {args.skip_vilt}")
    print("=" * 60)

    run_test("1. Imports",                test_imports)
    run_test("2. Dataset splits",         lambda: test_dataset(args.data_dir))
    run_test("3. Metrics",                test_metrics)
    run_test("4. CLIP frozen+concat fwd", lambda: test_clip_forward(args.data_dir))
    run_test("5. CLIP LoRA r=4 fwd",      lambda: test_clip_forward(args.data_dir, "lora", "concat", lora_r=4))
    run_test("6. CLIP unfreeze_1 fwd",    lambda: test_clip_forward(args.data_dir, "unfreeze_1", "concat"))
    run_test("7. CLIP frozen+xattn1 fwd", lambda: test_clip_forward(args.data_dir, "frozen", "crossattn1"))
    run_test("8. CLIP frozen+elemwise",   lambda: test_clip_forward(args.data_dir, "frozen", "elemwise"))
    run_test("9. CLIP frozen+gated",      lambda: test_clip_forward(args.data_dir, "frozen", "gated"))
    run_test("10. CLIP training step",    lambda: test_training_step_clip(args.data_dir))
    run_test("11. Checkpoint save/load",  lambda: test_checkpoint(args.data_dir))

    if not args.skip_vilt:
        run_test("12. ViLT full_ft fwd",   lambda: test_vilt_forward(args.data_dir, "full_ft"))
        run_test("13. ViLT LoRA r=4 fwd",  lambda: test_vilt_forward(args.data_dir, "lora", lora_r=4))
        run_test("14. ViLT unfreeze_3 fwd",lambda: test_vilt_forward(args.data_dir, "unfreeze_3"))
        run_test("15. ViLT training step", lambda: test_training_step_vilt(args.data_dir))
        run_test("16. Ablation modes",     lambda: test_ablation_modes(args.data_dir))
    else:
        run_test("12. Ablation modes (CLIP only)", lambda: test_ablation_modes(args.data_dir))

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"  {GREEN}ALL {total} TESTS PASSED{RESET} — можно запускать обучение!")
    else:
        print(f"  {RED}{failed} FAILED{RESET} / {passed} passed — исправь ошибки перед запуском.")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
