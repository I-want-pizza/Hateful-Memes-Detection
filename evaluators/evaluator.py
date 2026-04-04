"""
Evaluator with ablation-mode support.

Ablation modes:
  "multimodal"  — normal inference (both modalities)
  "text_only"   — image replaced with zeros / blank pixels
  "image_only"  — text replaced with empty / zero tokens
"""
import torch
from tqdm import tqdm

from utils.metrics import compute_confusion_matrix, compute_metrics


class Evaluator:
    def __init__(self, model, device: str = "cuda") -> None:
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        data_loader,
        desc: str = "Evaluation",
        ablation: str = "multimodal",
    ) -> dict:
        """
        Run inference and return metrics dict.

        ablation: "multimodal" | "text_only" | "image_only"
        """
        assert ablation in ("multimodal", "text_only", "image_only"), \
            f"Unknown ablation mode: '{ablation}'"

        # "text_only"  → keep only text  → zero out image embedding
        # "image_only" → keep only image → zero out text embedding
        text_mask  = (ablation == "image_only")   # image_only: keep image, zero text
        image_mask = (ablation == "text_only")    # text_only:  keep text, zero image

        self.model.eval()
        all_logits, all_labels = [], []

        for batch in tqdm(data_loader, desc=f"{desc} [{ablation}]"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(
                batch["pixel_values"],
                batch["input_ids"],
                batch["attention_mask"],
                text_mask=text_mask,
                image_mask=image_mask,
            )
            all_logits.append(logits.cpu())
            all_labels.append(batch["labels"].cpu())

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        metrics = compute_metrics(logits, labels)
        metrics["confusion_matrix"] = compute_confusion_matrix(logits, labels).tolist()
        return metrics
