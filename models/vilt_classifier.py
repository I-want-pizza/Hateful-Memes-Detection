"""
ViLT-based multimodal classifier (early fusion).

Supports full fine-tuning, LoRA, partial unfreezing,
and various regularization strategies.
"""
import torch
import torch.nn as nn
from transformers import ViltModel


class ViLTClassifier(nn.Module):
    """
    ViLT backbone + custom classification head.

    Config keys used:
        model.model_name        : HuggingFace model name
        model.num_classes       : number of output classes (2)
        model.hidden_dim        : intermediate size of classification head (default 768)
        model.dropout           : dropout in head (default 0.1)
        model.encoder_strategy  : "full_ft" | "lora" | "unfreeze_3" | "unfreeze_6"
        model.lora_r            : LoRA rank (only when encoder_strategy == "lora")
        model.lora_alpha        : LoRA alpha (only when encoder_strategy == "lora")
        model.lora_dropout      : LoRA dropout (default 0.1)
        model.gradient_checkpointing: bool (default False)
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]

        backbone_name    = mcfg["model_name"]
        num_classes      = mcfg["num_classes"]
        hidden_dim       = mcfg.get("hidden_dim", 768)
        dropout          = mcfg.get("dropout", 0.1)
        encoder_strategy = mcfg.get("encoder_strategy", "full_ft")
        grad_ckpt        = mcfg.get("gradient_checkpointing", False)

        # ── Load backbone ────────────────────────────────────────────────
        vilt_base = ViltModel.from_pretrained(backbone_name)

        if grad_ckpt:
            vilt_base.gradient_checkpointing_enable()

        # ── Apply encoder strategy ───────────────────────────────────────
        if encoder_strategy == "full_ft":
            self.vilt = vilt_base
            # All parameters trainable by default

        elif encoder_strategy == "lora":
            from peft import LoraConfig, get_peft_model  # type: ignore
            lora_cfg = LoraConfig(
                r=mcfg.get("lora_r", 8),
                lora_alpha=mcfg.get("lora_alpha", 16),
                lora_dropout=mcfg.get("lora_dropout", 0.1),
                target_modules=["query", "value"],
                bias="none",
            )
            self.vilt = get_peft_model(vilt_base, lora_cfg)

        elif encoder_strategy.startswith("unfreeze_"):
            n = int(encoder_strategy.split("_")[1])
            self.vilt = vilt_base
            # Freeze everything
            for p in self.vilt.parameters():
                p.requires_grad = False
            # Unfreeze last N transformer layers
            for layer in self.vilt.encoder.layer[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            # Unfreeze final norm and pooler
            for p in self.vilt.layernorm.parameters():
                p.requires_grad = True
            for p in self.vilt.pooler.parameters():
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown encoder_strategy: '{encoder_strategy}'")

        # ── Classification head ──────────────────────────────────────────
        input_dim = self.vilt.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_head_weights()

    def _init_head_weights(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    # ────────────────────────────────────────────────────────────────────
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        text_mask: bool = False,
        image_mask: bool = False,
    ) -> torch.Tensor:
        """
        text_mask / image_mask: replace inputs with blanks for ablation.

        For ViLT (unified transformer), ablation must happen at input level:
          - image_mask: replace pixel_values with zeros (black image)
          - text_mask:  replace input_ids with [CLS][SEP] (minimal text)
        """
        if image_mask:
            pixel_values = torch.zeros_like(pixel_values)
        if text_mask:
            # Keep [CLS]=101 and [SEP]=102 tokens, pad rest with 0
            blank = torch.zeros_like(input_ids)
            blank[:, 0] = 101  # [CLS]
            # Find length and put [SEP] at position 1
            blank[:, 1] = 102  # [SEP]
            input_ids = blank
            attention_mask = torch.zeros_like(attention_mask)
            attention_mask[:, 0] = 1
            attention_mask[:, 1] = 1

        outputs = self.vilt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_dict=True,
        )
        pooled = outputs.pooler_output   # (B, 768)
        return self.classifier(pooled)

    # ────────────────────────────────────────────────────────────────────
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
