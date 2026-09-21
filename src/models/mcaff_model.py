"""
MCAFF — Multi-Head Cross-Attention Feature Fusion for Dynamic API Resolution Detection.

Novelty: Thay vì concat/gated fusion thông thường, MCAFF xem mỗi nhóm feature
(Static, Behavior, Graph-derived) như một "token" và áp dụng Multi-Head Cross-Attention
để capture tương tác liên nhóm — cho phép mô hình tự học mối quan hệ giữa
static indicators (IAT) và dynamic behavior (API call patterns).

Architecture:
  Static PE [static_dim] ──→ Linear Projection ──→ token_1 [d_model]
  Behavior  [behavior_dim] ──→ Linear Projection ──→ token_2 [d_model]  ──→ Cross-Attention ──→ FFN ──→ Classifier
  Graph-derived [graph_dim] ──→ Linear Projection ──→ token_3 [d_model]

Reference: Vaswani et al. "Attention Is All You Need" (2017) — adapted for
feature-group-level cross-attention in malware analysis.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGroupProjection(nn.Module):
    """Project each feature group to a common d_model dimension."""

    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionBlock(nn.Module):
    """
    Transformer-style block with Multi-Head Self-Attention over feature group tokens.

    Input: [batch, num_groups, d_model] — mỗi group là 1 token.
    Output: [batch, num_groups, d_model] — sau cross-attention + FFN.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Self-Attention with residual + LayerNorm
        attn_out, self.last_attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-Forward Network with residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class MCAFF(nn.Module):
    """
    Multi-Head Cross-Attention Feature Fusion (MCAFF) for G1/G2/G3 classification.

    Kiến trúc novel: 3 feature groups (Static, Behavior, Graph-derived) được project
    thành tokens, cross-attend lẫn nhau qua Transformer layers, rồi aggregate
    để phân loại kỹ thuật Dynamic API Resolution.

    Args:
        static_dim: Dimension of static PE features (default 64)
        behavior_dim: Dimension of behavior statistics features (default 15)
        graph_dim: Dimension of graph-derived features (default 5)
        d_model: Common projection dimension for all groups (default 64)
        num_heads: Number of attention heads (default 4)
        num_attention_layers: Number of cross-attention layers (default 2)
        ffn_ratio: FFN hidden dim = d_model * ffn_ratio (default 4)
        num_classes: Number of DAR technique groups (default 3: G1/G2/G3)
        num_families: Number of malware families for auxiliary task
        family_classifier: Whether to use auxiliary family classifier
        dropout: Dropout rate
        pool_strategy: How to aggregate tokens — 'mean', 'cls', or 'concat'
    """

    def __init__(
        self,
        static_dim: int = 64,
        behavior_dim: int = 15,
        graph_dim: int = 5,
        d_model: int = 64,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        ffn_ratio: int = 4,
        num_classes: int = 3,
        num_families: int = 20,
        family_classifier: bool = True,
        dropout: float = 0.1,
        pool_strategy: Literal["mean", "cls", "concat"] = "mean",
    ):
        super().__init__()
        self.pool_strategy = pool_strategy
        self.num_groups = 3

        # Feature group projections
        self.static_proj = FeatureGroupProjection(static_dim, d_model, dropout)
        self.behavior_proj = FeatureGroupProjection(behavior_dim, d_model, dropout)
        self.graph_proj = FeatureGroupProjection(graph_dim, d_model, dropout)

        # Learnable group type embeddings (like segment embeddings in BERT)
        self.group_embeddings = nn.Embedding(self.num_groups, d_model)

        # Cross-Attention layers
        self.attention_layers = nn.ModuleList([
            CrossAttentionBlock(d_model, num_heads, ffn_ratio, dropout)
            for _ in range(num_attention_layers)
        ])

        # Classification heads
        if pool_strategy == "concat":
            classifier_dim = d_model * self.num_groups
        else:
            classifier_dim = d_model

        self.group_head = nn.Sequential(
            nn.Linear(classifier_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self.family_classifier = family_classifier
        if family_classifier:
            self.family_head = nn.Sequential(
                nn.Linear(classifier_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_families),
            )

    def forward(
        self,
        static_x: torch.Tensor,
        behavior_x: torch.Tensor,
        graph_x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            static_x:   [batch, static_dim]   — Static PE/IAT features
            behavior_x: [batch, behavior_dim] — Behavior statistics
            graph_x:    [batch, graph_dim]    — Graph-derived statistics

        Returns:
            dict with 'logits_group' and optionally 'logits_family', 'attn_weights'
        """
        # Project each group to d_model
        h_static = self.static_proj(static_x)       # [batch, d_model]
        h_behavior = self.behavior_proj(behavior_x)  # [batch, d_model]
        h_graph = self.graph_proj(graph_x)           # [batch, d_model]

        # Stack as token sequence: [batch, 3, d_model]
        tokens = torch.stack([h_static, h_behavior, h_graph], dim=1)

        # Add learnable group type embeddings
        group_ids = torch.arange(self.num_groups, device=tokens.device)
        tokens = tokens + self.group_embeddings(group_ids).unsqueeze(0)

        # Apply cross-attention layers
        for attn_layer in self.attention_layers:
            tokens = attn_layer(tokens)

        # Aggregate tokens
        if self.pool_strategy == "mean":
            fused = tokens.mean(dim=1)  # [batch, d_model]
        elif self.pool_strategy == "concat":
            fused = tokens.reshape(tokens.size(0), -1)  # [batch, 3*d_model]
        else:  # cls — use first token
            fused = tokens[:, 0, :]  # [batch, d_model]

        # Classification
        logits_group = self.group_head(fused)
        out: dict[str, torch.Tensor] = {"logits_group": logits_group}

        if self.family_classifier:
            out["logits_family"] = self.family_head(fused)

        # Store attention weights for interpretability (research paper visualization)
        attn_weights = []
        for layer in self.attention_layers:
            if hasattr(layer, 'last_attn_weights') and layer.last_attn_weights is not None:
                attn_weights.append(layer.last_attn_weights.detach())
        if attn_weights:
            out["attn_weights"] = torch.stack(attn_weights)

        return out

    def get_attention_map(self) -> list[torch.Tensor]:
        """Extract attention maps for visualization in research paper."""
        maps = []
        for layer in self.attention_layers:
            if hasattr(layer, 'last_attn_weights') and layer.last_attn_weights is not None:
                maps.append(layer.last_attn_weights.detach().cpu())
        return maps
