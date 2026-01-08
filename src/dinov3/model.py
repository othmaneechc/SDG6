from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_patch_size(backbone) -> int:
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "patch_size"):
        ps = backbone.patch_embed.patch_size
        if isinstance(ps, (tuple, list)):
            return ps[0]
        return int(ps)
    return 16


def masked_global_pool(tokens: torch.Tensor, mask: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    tokens: [B, N, C] patch tokens
    mask: [B, H, W] binary mask (0/1) matching image space, already padded
    patch_size: patch size of the backbone (e.g., 16)
    """
    b, n, c = tokens.shape
    h = mask.shape[-2] // patch_size
    w = mask.shape[-1] // patch_size
    if h * w != n:
        raise ValueError(f"Token count {n} does not match mask grid {h}x{w}")
    mask_ds = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest")  # [B,1,h,w]
    flat = mask_ds.flatten(2)  # [B,1,N]
    weights = flat / (flat.sum(dim=2, keepdim=True) + 1e-6)
    pooled = torch.bmm(weights, tokens).squeeze(1)  # [B,C]
    return pooled


class Dinov3HFBackbone(nn.Module):
    """Wrap an HF DINOv3 model to expose patch tokens like the original torch.hub interface."""

    def __init__(self, model: nn.Module, patch_size: int | None = None):
        super().__init__()
        self.model = model
        hidden = getattr(model.config, "hidden_size", None) or getattr(model.config, "embed_dim", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden size from HF model config")
        self.embed_dim = hidden
        ps = patch_size if patch_size is not None else getattr(model.config, "patch_size", None)
        if isinstance(ps, (tuple, list)):
            ps = ps[0]
        self.patch_embed = SimpleNamespace(patch_size=int(ps) if ps is not None else 16)
        self.blocks = getattr(model, "blocks", None)
        if self.blocks is None:
            encoder = getattr(model, "encoder", None)
            self.blocks = getattr(encoder, "layers", None) or getattr(encoder, "layer", None)
        self.norm = getattr(model, "norm", None) or getattr(model, "layernorm", None) or getattr(
            model, "layer_norm", None
        )

    def forward_features(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        if tokens is None:
            raise ValueError("HF Dinov3 model did not return last_hidden_state")
        expected = (pixel_values.shape[-2] // self.patch_embed.patch_size) * (
            pixel_values.shape[-1] // self.patch_embed.patch_size
        )
        if tokens.shape[1] < expected:
            raise ValueError(f"Unexpected token count {tokens.shape[1]} (expected at least {expected})")
        # Some HF Dinov3 models include CLS + register tokens; keep only the trailing patch tokens.
        tokens = tokens[:, -expected:, :]
        return {"x_norm_patchtokens": tokens}


class DinoPairedClassifier(nn.Module):
    """
    Shared DINO backbone on before/after images, masked pooling, then delta head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden_dim: int = 512,
        composition: str = "b_a_delta",
    ):
        super().__init__()
        self.backbone = backbone
        self.composition = composition
        dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", None))
        if dim is None:
            raise ValueError("Cannot infer feature dim from backbone")
        self.patch_size = detect_patch_size(backbone)
        self.token_delta = composition in {"delta_tokens", "delta_tokens_abs"}

        if self.token_delta:
            token_dim = dim if composition == "delta_tokens" else dim * 2
            self.delta_proj = nn.Linear(token_dim, dim)
            self.pool = MaskedTokenPool(dim)
            head_dim = dim
        else:
            feat_dim = self._feature_dim(dim)
            head_dim = feat_dim

        self.head = nn.Sequential(nn.Linear(head_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden_dim, num_classes))

    def _feature_dim(self, dim: int) -> int:
        if self.composition == "b_a_delta":
            return dim * 3
        if self.composition == "delta_only":
            return dim
        if self.composition == "delta_abs":
            return dim * 2
        if self.composition == "b_a_delta_abs":
            return dim * 4
        if self.composition in {"delta_tokens", "delta_tokens_abs"}:
            return dim
        raise ValueError(f"Unknown composition '{self.composition}'")

    def _compose(self, pooled_b: torch.Tensor, pooled_a: torch.Tensor) -> torch.Tensor:
        delta = pooled_a - pooled_b
        if self.composition == "b_a_delta":
            parts = (pooled_b, pooled_a, delta)
        elif self.composition == "delta_only":
            parts = (delta,)
        elif self.composition == "delta_abs":
            parts = (delta, delta.abs())
        elif self.composition == "b_a_delta_abs":
            parts = (pooled_b, pooled_a, delta, delta.abs())
        elif self.composition in {"delta_tokens", "delta_tokens_abs"}:
            raise ValueError("Token compositions are handled separately")
        else:
            raise ValueError(f"Unknown composition '{self.composition}'")
        return torch.cat(parts, dim=1)

    def _mask_to_tokens(self, mask_before: torch.Tensor, mask_after: torch.Tensor, n_tokens: int) -> torch.Tensor:
        h = mask_before.shape[-2] // self.patch_size
        w = mask_before.shape[-1] // self.patch_size
        if h * w != n_tokens:
            raise ValueError(f"Token count {n_tokens} does not match mask grid {h}x{w}")
        mask_union = torch.maximum(mask_before, mask_after)
        mask_ds = F.interpolate(mask_union.unsqueeze(1), size=(h, w), mode="nearest")
        mask_flat = (mask_ds.flatten(1) > 0.0)
        return mask_flat

    def forward(
        self,
        img_before: torch.Tensor,
        mask_before: torch.Tensor,
        img_after: torch.Tensor,
        mask_after: torch.Tensor,
    ) -> torch.Tensor:
        feats_b = self.backbone.forward_features(img_before)
        feats_a = self.backbone.forward_features(img_after)
        tokens_b = feats_b.get("x_norm_patchtokens") if isinstance(feats_b, dict) else None
        tokens_a = feats_a.get("x_norm_patchtokens") if isinstance(feats_a, dict) else None
        if tokens_b is None or tokens_a is None:
            raise ValueError("Backbone forward_features did not return patch tokens")

        if self.token_delta:
            delta_tokens = tokens_a - tokens_b
            if self.composition == "delta_tokens_abs":
                delta_tokens = torch.cat([delta_tokens, delta_tokens.abs()], dim=-1)
            delta_tokens = self.delta_proj(delta_tokens)
            mask_tokens = self._mask_to_tokens(mask_before, mask_after, delta_tokens.shape[1])
            pooled = self.pool(delta_tokens, mask_tokens)
            return self.head(pooled)

        pooled_b = masked_global_pool(tokens_b, mask_before, self.patch_size)
        pooled_a = masked_global_pool(tokens_a, mask_after, self.patch_size)
        features = self._compose(pooled_b, pooled_a)
        return self.head(features)


class MaskedTokenPool(nn.Module):
    """Attention-style pooling over patch tokens with an optional binary mask."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: [B, N, D], mask: [B, N] bool
        scores = self.score(tokens).squeeze(-1)  # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        attn = torch.softmax(scores, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), tokens).squeeze(1)
        return pooled
