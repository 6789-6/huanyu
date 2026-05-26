"""Streaming keyword recognizer for the CE-CSL software MVP path."""
from collections import Counter
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset_keypoint import KP_DIM
from model_keyword import KeywordPrediction, topk_keywords
from train_keyword_classifier import load_keyword_checkpoint


DEFAULT_DEMO_WHITELIST = {"我", "你", "我们", "这里", "这", "今天"}


class KeywordStabilizer:
    """Filter keyword predictions for demos using whitelist + repeated hits."""

    def __init__(
        self,
        whitelist: set[str] | None = None,
        min_hits: int = 2,
        cooldown_steps: int = 2,
    ):
        self.whitelist = set(whitelist) if whitelist is not None else None
        self.min_hits = max(1, min_hits)
        self.cooldown_steps = max(0, cooldown_steps)
        self.counts: Counter[str] = Counter()
        self.cooldowns: dict[str, int] = {}

    def update(self, predictions: list[KeywordPrediction]) -> list[KeywordPrediction]:
        for token in list(self.cooldowns):
            self.cooldowns[token] -= 1
            if self.cooldowns[token] <= 0:
                del self.cooldowns[token]

        present: dict[str, KeywordPrediction] = {}
        for pred in predictions:
            if self.whitelist is not None and pred.token not in self.whitelist:
                continue
            present[pred.token] = pred

        for token in list(self.counts):
            if token not in present:
                del self.counts[token]
        for token in present:
            self.counts[token] += 1

        accepted: list[KeywordPrediction] = []
        for token, pred in present.items():
            if self.counts[token] >= self.min_hits and token not in self.cooldowns:
                accepted.append(pred)
                self.cooldowns[token] = self.cooldown_steps
                self.counts[token] = 0
        accepted.sort(key=lambda pred: pred.confidence, reverse=True)
        return accepted

    def reset(self) -> None:
        self.counts.clear()
        self.cooldowns.clear()


def compose_demo_sentence(tokens: list[str]) -> str:
    """Compose a short display sentence for the stable demo whitelist."""
    normalized = set(tokens)
    if {"我", "这里"}.issubset(normalized) or {"ME", "HERE"}.issubset(normalized):
        return "我在这里"
    if {"你", "今天"}.issubset(normalized) or {"YOU", "TODAY"}.issubset(normalized):
        return "你今天"
    mapping = {
        "我": "我",
        "你": "你",
        "我们": "我们",
        "这里": "这里",
        "这": "这",
        "今天": "今天",
        "ME": "我",
        "YOU": "你",
        "WE": "我们",
        "HERE": "这里",
        "THIS": "这",
        "TODAY": "今天",
    }
    return "".join(mapping.get(token, token) for token in tokens)


class StreamingKeywordRecognizer:
    """Run multi-label keyword classification on a sliding keypoint window."""

    def __init__(
        self,
        model: nn.Module,
        idx_to_token: dict[int, str],
        window_size: int = 64,
        stride: int = 8,
        confidence_threshold: float = 0.5,
        top_k: int = 5,
        stabilizer: KeywordStabilizer | None = None,
    ):
        self.model = model.eval()
        self.idx_to_token = idx_to_token
        self.window_size = window_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.stabilizer = stabilizer
        self.buffer: deque[torch.Tensor] = deque(maxlen=window_size)
        self.frame_count = 0
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
        window_size: int = 64,
        stride: int = 8,
        confidence_threshold: float | None = None,
        top_k: int = 5,
        stabilize: bool = False,
        whitelist: set[str] | None = None,
        min_hits: int = 2,
        cooldown_steps: int = 2,
    ) -> "StreamingKeywordRecognizer":
        model, meta = load_keyword_checkpoint(path)
        model.to(torch.device(device))
        threshold = meta.get("best_threshold", 0.5) if confidence_threshold is None else confidence_threshold
        stabilizer = KeywordStabilizer(
            whitelist=DEFAULT_DEMO_WHITELIST if whitelist is None else whitelist,
            min_hits=min_hits,
            cooldown_steps=cooldown_steps,
        ) if stabilize else None
        return cls(
            model=model,
            idx_to_token=meta["idx_to_token"],
            window_size=window_size,
            stride=stride,
            confidence_threshold=threshold,
            top_k=top_k,
            stabilizer=stabilizer,
        )

    def _ready(self) -> bool:
        return (
            len(self.buffer) == self.window_size
            and self.frame_count > 0
            and self.frame_count % self.stride == 0
        )

    @torch.no_grad()
    def add_frame(self, keypoints: np.ndarray) -> list[KeywordPrediction] | None:
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.shape != (KP_DIM,):
            raise ValueError(f"Expected one keypoint frame with shape ({KP_DIM},), got {keypoints.shape}")
        self.buffer.append(torch.from_numpy(keypoints))
        self.frame_count += 1
        if not self._ready():
            return None

        seq = torch.stack(list(self.buffer), dim=0).unsqueeze(0).to(self.device)
        valid_mask = torch.ones(1, seq.size(1), dtype=torch.bool, device=self.device)
        logits = self.model(seq, valid_mask)
        preds = topk_keywords(
            logits,
            self.idx_to_token,
            k=self.top_k,
            threshold=self.confidence_threshold,
        )[0]
        if self.stabilizer is not None:
            preds = self.stabilizer.update(preds)
        return preds or None

    def reset(self) -> None:
        self.buffer.clear()
        self.frame_count = 0
        if self.stabilizer is not None:
            self.stabilizer.reset()
