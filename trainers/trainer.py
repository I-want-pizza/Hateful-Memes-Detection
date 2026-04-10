"""
Universal trainer with:
  - wandb integration (metrics, config, artifacts, system stats)
  - CSV epoch logging
  - Mixed-precision (FP16) via torch.amp
  - Gradient accumulation
  - Early stopping on ROC-AUC
  - Discriminative learning rates (for unfreeze experiments)
"""
import csv
import subprocess
import time
from pathlib import Path

import torch
import torch.amp as amp
from tqdm import tqdm

import wandb

from config.settings import settings
from utils.checkpoint import save_checkpoint
from utils.metrics import compute_metrics


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, cfg, logger):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.device       = device
        self.cfg          = cfg
        self.logger       = logger

        tcfg = cfg["training"]
        self.epochs      = int(tcfg["epochs"])
        self.use_fp16    = bool(tcfg.get("fp16", False))
        self.accum_steps = int(tcfg.get("grad_accumulation_steps", 1))
        self.patience    = int(tcfg.get("early_stopping_patience", 10))
        self.output_dir  = Path(cfg["experiment"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        label_smoothing  = float(tcfg.get("label_smoothing", 0.0))
        self.criterion   = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        amp_device       = device if device == "cuda" else "cpu"
        self.scaler      = amp.GradScaler(amp_device, enabled=self.use_fp16 and device == "cuda")

        # CSV logging
        self.csv_path = self.output_dir / "metrics.csv"
        self._csv_header_written = False

        # wandb — API key из settings (.env), project из конфига или settings
        if settings.wandb_api_key:
            import os
            os.environ["WANDB_API_KEY"] = settings.wandb_api_key
        lcfg = cfg.get("logging", {})
        wandb.init(
            project = lcfg.get("wandb_project", settings.wandb_project),
            name    = cfg["experiment"]["name"],
            config  = cfg,
            dir     = str(self.output_dir),
            reinit  = True,
        )
        self._log_system_info()

    # ── System info ───────────────────────────────────────────────────────
    def _log_system_info(self) -> None:
        try:
            sysinfo = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True
            ).stdout
        except FileNotFoundError:
            sysinfo = "nvidia-smi not available"

        trainable = self.model.count_trainable_params() if hasattr(self.model, "count_trainable_params") else -1
        total      = sum(p.numel() for p in self.model.parameters())

        info_path = self.output_dir / "sysinfo.txt"
        with open(info_path, "w") as f:
            f.write(sysinfo)
            f.write(f"\nTrainable params : {trainable:,}\n")
            f.write(f"Total params     : {total:,}\n")

        wandb.run.summary["trainable_params"] = trainable
        wandb.run.summary["total_params"]     = total
        self.logger.info(f"Trainable params: {trainable:,} / {total:,}")

    # ── CSV helper ────────────────────────────────────────────────────────
    def _write_csv_row(self, row: dict) -> None:
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    # ── Training epoch ────────────────────────────────────────────────────
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_loader, desc="Train", leave=False)):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with amp.autocast(self.device, enabled=self.use_fp16 and self.device == "cuda"):
                logits = self.model(
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                loss = self.criterion(logits, batch["labels"]) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps

        return total_loss / len(self.train_loader)

    # ── Validation epoch ──────────────────────────────────────────────────
    @torch.no_grad()
    def eval_epoch(self) -> dict:
        self.model.eval()
        all_logits, all_labels = [], []

        with amp.autocast(self.device, enabled=self.use_fp16 and self.device == "cuda"):
            for batch in tqdm(self.val_loader, desc="Val", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                all_logits.append(logits.cpu())
                all_labels.append(batch["labels"].cpu())

        return compute_metrics(torch.cat(all_logits), torch.cat(all_labels))

    # ── Main loop ─────────────────────────────────────────────────────────
    def train(self) -> None:
        best_auroc    = 0.0
        patience_cnt  = 0
        best_ckpt     = str(self.output_dir / "best.pt")

        self.logger.info(
            f"Training: fp16={self.use_fp16}, accum={self.accum_steps}, "
            f"epochs={self.epochs}, patience={self.patience}"
        )

        for epoch in range(1, self.epochs + 1):
            t0         = time.time()
            train_loss = self.train_epoch()
            metrics    = self.eval_epoch()
            elapsed    = time.time() - t0

            # GPU memory
            gpu_alloc = gpu_reserved = 0
            if torch.cuda.is_available():
                gpu_alloc   = torch.cuda.memory_allocated()  / 1e6
                gpu_reserved = torch.cuda.memory_reserved() / 1e6

            # Log
            log_msg = (
                f"Epoch {epoch:>3}/{self.epochs} | "
                f"Loss {train_loss:.4f} | "
                f"AUC {metrics['auroc']:.4f} | "
                f"Acc {metrics['accuracy']:.4f} | "
                f"F1 {metrics['f1']:.4f} | "
                f"P {metrics['precision']:.4f} | "
                f"R {metrics['recall']:.4f} | "
                f"{elapsed:.0f}s"
            )
            self.logger.info(log_msg)

            # wandb
            wandb.log({
                "epoch":             epoch,
                "train_loss":        train_loss,
                "val_auroc":         metrics["auroc"],
                "val_accuracy":      metrics["accuracy"],
                "val_f1":            metrics["f1"],
                "val_precision":     metrics["precision"],
                "val_recall":        metrics["recall"],
                "gpu_alloc_mb":      gpu_alloc,
                "gpu_reserved_mb":   gpu_reserved,
                "epoch_time_s":      elapsed,
            })

            # CSV
            self._write_csv_row({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                **{f"val_{k}": round(v, 6) for k, v in metrics.items()},
                "epoch_time_s": round(elapsed, 1),
            })

            # Checkpoint + early stopping
            if metrics["auroc"] > best_auroc:
                best_auroc   = metrics["auroc"]
                patience_cnt = 0
                save_checkpoint(self.model, self.optimizer, epoch, metrics, self.cfg, best_ckpt)
                wandb.run.summary["best_auroc"] = best_auroc
                wandb.run.summary["best_epoch"] = epoch
                self.logger.info(f"  ✓ New best AUC: {best_auroc:.4f} — checkpoint saved")
            else:
                patience_cnt += 1
                self.logger.info(f"  No improvement ({patience_cnt}/{self.patience})")
                if patience_cnt >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # Upload only the final best checkpoint as a single wandb artifact
        if Path(best_ckpt).exists():
            artifact = wandb.Artifact(
                name=f"model-{self.cfg['experiment']['name']}",
                type="model",
            )
            artifact.add_file(best_ckpt)
            wandb.log_artifact(artifact)

        wandb.finish()
        self.logger.info(f"Training complete. Best Val AUC: {best_auroc:.4f}")
