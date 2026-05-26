"""Lightweight multi-label keyword classifier for CE-CSL keypoints."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KeywordPrediction:
    token: str
    confidence: float
    index: int


class KeywordClassifier(nn.Module):
    """Small GRU encoder with a sigmoid multi-label head."""

    def __init__(
        self,
        kp_dim: int = 258,
        num_keywords: int = 100,
        d_model: int = 96,
        hidden_size: int = 96,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.kp_dim = kp_dim
        self.num_keywords = num_keywords
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.dropout_p = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(kp_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.encoder = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_keywords),
        )

    def forward(self, keypoints: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(keypoints)
        out, _ = self.encoder(x)
        if valid_mask is None:
            pooled = out.mean(dim=1)
        else:
            mask = valid_mask.to(dtype=out.dtype, device=out.device).unsqueeze(-1)
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.head(pooled)

    def config(self) -> dict:
        return {
            "kp_dim": self.kp_dim,
            "num_keywords": self.num_keywords,
            "d_model": self.d_model,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout_p,
        }


@torch.no_grad()
def topk_keywords(
    logits: torch.Tensor,
    idx_to_token: dict[int, str],
    k: int = 10,
    threshold: float | None = None,
) -> list[list[KeywordPrediction]]:
    probs = torch.sigmoid(logits)
    k = min(k, probs.size(-1))
    confs, ids = probs.topk(k, dim=-1)
    batches: list[list[KeywordPrediction]] = []
    for row_confs, row_ids in zip(confs, ids):
        preds: list[KeywordPrediction] = []
        for conf, idx in zip(row_confs.cpu(), row_ids.cpu()):
            score = float(conf)
            if threshold is not None and score < threshold:
                continue
            int_idx = int(idx)
            preds.append(KeywordPrediction(
                token=idx_to_token.get(int_idx, str(int_idx)),
                confidence=score,
                index=int_idx,
            ))
        batches.append(preds)
    return batches
