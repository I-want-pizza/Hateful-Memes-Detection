"""
CLIP-based multimodal classifier.

Supports all fusion strategies (concat, elemwise, gated, crossattn)
and all encoder adaptation strategies (frozen, lora, unfreeze_N).
"""
import torch
import torch.nn as nn
from transformers import CLIPModel


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class CLIPClassifier(nn.Module):
    """
    CLIP encoder + configurable fusion head.

    Config keys used:
        model.clip_model        : HuggingFace model name
        model.num_classes       : number of output classes (2)
        model.dropout           : dropout in classification head
        model.encoder_strategy  : "frozen" | "lora" | "unfreeze_1" | "unfreeze_3" | "unfreeze_6"
        model.fusion            : "concat" | "elemwise" | "gated" | "crossattn1" | "crossattn2"
        model.lora_r            : LoRA rank (only when encoder_strategy == "lora")
        model.lora_alpha        : LoRA alpha (only when encoder_strategy == "lora")
        model.lora_dropout      : LoRA dropout (default 0.1)
    """

    EMBED_DIM = 512  # CLIP ViT-B/32 output dimensionality

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        mcfg = cfg["model"]

        backbone_name    = mcfg["clip_model"]
        num_classes      = mcfg["num_classes"]
        dropout          = mcfg.get("dropout", 0.2)
        encoder_strategy = mcfg.get("encoder_strategy", "frozen")
        fusion           = mcfg.get("fusion", "concat")

        self.fusion = fusion
        d = self.EMBED_DIM

        # ── Load backbone ────────────────────────────────────────────────
        self.clip = CLIPModel.from_pretrained(backbone_name)

        # ── Apply encoder strategy ───────────────────────────────────────
        if encoder_strategy == "frozen":
            for p in self.clip.parameters():
                p.requires_grad = False

        elif encoder_strategy == "lora":
            # Lazy import so peft is only required when actually used
            from peft import LoraConfig, get_peft_model  # type: ignore
            lora_cfg = LoraConfig(
                r=mcfg.get("lora_r", 8),
                lora_alpha=mcfg.get("lora_alpha", 16),
                lora_dropout=mcfg.get("lora_dropout", 0.1),
                target_modules=["q_proj", "v_proj"],
                bias="none",
            )
            self.clip = get_peft_model(self.clip, lora_cfg)

        elif encoder_strategy.startswith("unfreeze_"):
            n = int(encoder_strategy.split("_")[1])
            # Freeze everything first
            for p in self.clip.parameters():
                p.requires_grad = False
            # Unfreeze last N transformer blocks in both encoders
            for layer in self.clip.vision_model.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.vision_model.post_layernorm.parameters():
                p.requires_grad = True
            for layer in self.clip.text_model.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            for p in self.clip.text_model.final_layer_norm.parameters():
                p.requires_grad = True
            # Always unfreeze projection heads
            for p in self.clip.visual_projection.parameters():
                p.requires_grad = True
            for p in self.clip.text_projection.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown encoder_strategy: '{encoder_strategy}'")

        # ── Build fusion head ────────────────────────────────────────────
        if fusion == "concat":
            self.classifier = _build_mlp(d * 2, d, num_classes, dropout)

        elif fusion == "elemwise":
            self.classifier = _build_mlp(d, d // 2, num_classes, dropout)

        elif fusion == "gated":
            self.gate = nn.Linear(d * 2, d)
            self.classifier = _build_mlp(d, d // 2, num_classes, dropout)

        elif fusion in ("crossattn1", "crossattn2"):
            n_layers = 1 if fusion == "crossattn1" else 2
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d, nhead=8, dim_feedforward=1024,
                dropout=dropout, batch_first=True,
            )
            self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
            # concat(attn_out, txt_emb) → MLP
            self.classifier = _build_mlp(d * 2, d, num_classes, dropout)

        else:
            raise ValueError(f"Unknown fusion method: '{fusion}'")

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
        text_mask / image_mask: zero out the respective embedding
        (used for modality ablation studies).
        """
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        img_emb = outputs.image_embeds   # (B, 512)
        txt_emb = outputs.text_embeds    # (B, 512)

        # Ablation: replace with zeros to isolate one modality
        if image_mask:
            img_emb = torch.zeros_like(img_emb)
        if text_mask:
            txt_emb = torch.zeros_like(txt_emb)

        # Fusion
        if self.fusion == "concat":
            fused = torch.cat([img_emb, txt_emb], dim=1)

        elif self.fusion == "elemwise":
            fused = img_emb * txt_emb

        elif self.fusion == "gated":
            g = torch.sigmoid(self.gate(torch.cat([img_emb, txt_emb], dim=1)))
            fused = g * img_emb + (1.0 - g) * txt_emb

        elif self.fusion in ("crossattn1", "crossattn2"):
            q  = img_emb.unsqueeze(1)    # (B, 1, 512) — query: image
            kv = txt_emb.unsqueeze(1)    # (B, 1, 512) — key/value: text
            attn_out = self.cross_attn(q, kv).squeeze(1)  # (B, 512)
            fused = torch.cat([attn_out, txt_emb], dim=1)

        return self.classifier(fused)

    # ────────────────────────────────────────────────────────────────────
    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
