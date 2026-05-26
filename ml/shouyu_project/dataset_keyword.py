"""Keyword-level multi-label dataset built from existing CE-CSL keypoints.

This avoids the unstable large-vocabulary CTC path.  Each CE-CSL sentence is
converted into a multi-hot vector over the most common gloss tokens, so the
software MVP can recognize useful keywords without recording new video.
"""
from collections import Counter
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from dataset_keypoint import (
    BLANK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    KeypointDataset,
    NUM_FRAMES,
)


DEFAULT_EXCLUDE_TOKENS = {
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    BLANK_TOKEN,
    "",
    ".",
    ",",
    "?",
    "!",
    ";",
    ":",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    '"',
    "'",
    "。",
    "，",
    "？",
    "！",
    "、",
    "；",
    "：",
    "（",
    "）",
    "《",
    "》",
    "“",
    "”",
}


def split_gloss(gloss: str) -> list[str]:
    """Split CE-CSL slash-separated gloss text into cleaned tokens."""
    return [token.strip() for token in str(gloss).split("/") if token.strip()]


def build_keyword_vocab(
    samples: Iterable[dict],
    top_k: int = 100,
    exclude_tokens: Iterable[str] | None = None,
    min_freq: int = 1,
) -> dict[str, int]:
    """Build a deterministic top-K gloss-token vocabulary."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    excluded = set(DEFAULT_EXCLUDE_TOKENS if exclude_tokens is None else exclude_tokens)

    counter: Counter[str] = Counter()
    for sample in samples:
        for token in split_gloss(sample.get("gloss", "")):
            if token not in excluded:
                counter[token] += 1

    ranked = sorted(
        ((token, count) for token, count in counter.items() if count >= min_freq),
        key=lambda item: (-item[1], item[0]),
    )
    return {token: idx for idx, (token, _) in enumerate(ranked[:top_k])}


class KeywordKeypointDataset(Dataset):
    """Wrap KeypointDataset and return multi-hot keyword labels."""

    def __init__(
        self,
        split: str = "train",
        base_dataset: Dataset | None = None,
        token_to_idx: dict[str, int] | None = None,
        top_k: int = 100,
        exclude_tokens: Iterable[str] | None = None,
        min_freq: int = 1,
        drop_empty: bool = True,
        augment: bool = False,
        num_frames: int | None = NUM_FRAMES,
    ):
        if base_dataset is None:
            base_dataset = KeypointDataset(split=split, augment=augment, num_frames=num_frames)
        if not hasattr(base_dataset, "samples"):
            raise ValueError("base_dataset must expose a samples list with gloss fields")

        self.split = split
        self.base_dataset = base_dataset
        self.token_to_idx = dict(token_to_idx) if token_to_idx is not None else build_keyword_vocab(
            base_dataset.samples,
            top_k=top_k,
            exclude_tokens=exclude_tokens,
            min_freq=min_freq,
        )
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.drop_empty = drop_empty

        self.samples: list[dict] = []
        self.base_indices: list[int] = []
        for base_idx, sample in enumerate(base_dataset.samples):
            tokens = split_gloss(sample.get("gloss", ""))
            selected = sorted({self.token_to_idx[token] for token in tokens if token in self.token_to_idx})
            if drop_empty and not selected:
                continue
            self.samples.append({
                "base_idx": base_idx,
                "gloss": sample.get("gloss", ""),
                "gloss_tokens": tokens,
                "label_indices": selected,
            })
            self.base_indices.append(base_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        base_item = self.base_dataset[sample["base_idx"]]
        labels = torch.zeros(len(self.token_to_idx), dtype=torch.float32)
        if sample["label_indices"]:
            labels[torch.tensor(sample["label_indices"], dtype=torch.long)] = 1.0

        return {
            "keypoints": base_item["keypoints"],
            "valid_mask": base_item["valid_mask"],
            "labels": labels,
            "gloss_tokens": list(sample["gloss_tokens"]),
            "gloss": sample["gloss"],
            "base_idx": sample["base_idx"],
            "path": base_item.get("path", ""),
        }


def collate_keyword_batch(batch: list[dict]) -> dict:
    """Pad variable-length keypoints and stack multi-hot labels."""
    max_len = max(item["keypoints"].shape[0] for item in batch)
    kp_dim = batch[0]["keypoints"].shape[1]
    keypoints = torch.zeros(len(batch), max_len, kp_dim, dtype=batch[0]["keypoints"].dtype)
    valid_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in batch], dim=0)

    for row, item in enumerate(batch):
        length = item["keypoints"].shape[0]
        keypoints[row, :length] = item["keypoints"]
        valid_masks[row, :length] = item["valid_mask"]

    return {
        "keypoints": keypoints,
        "valid_mask": valid_masks,
        "input_lengths": valid_masks.sum(dim=1).long(),
        "labels": labels,
        "gloss_tokens": [item["gloss_tokens"] for item in batch],
        "gloss": [item.get("gloss", "") for item in batch],
        "paths": [item.get("path", "") for item in batch],
        "base_indices": torch.tensor([item["base_idx"] for item in batch], dtype=torch.long),
    }
