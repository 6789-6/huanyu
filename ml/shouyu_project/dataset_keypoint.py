"""Keypoint dataset: loads pre-extracted MediaPipe Holistic keypoints for CSLR.

Keypoint format per frame (258-dim):
  pose      33 × 4 (x, y, z, visibility) = 132
  left_hand 21 × 3 (x, y, z)            =  63
  right_hand 21 × 3 (x, y, z)           =  63
"""
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(os.environ.get("CE_CSL_ROOT", "data/CE-CSL"))
SHOUYU_ROOT = Path(__file__).resolve().parent
KEYPOINT_DIR = SHOUYU_ROOT / "keypoints"

NUM_FRAMES = 64
POSE_DIM = 33 * 4     # 132
HAND_DIM = 21 * 3     # 63
KP_DIM = POSE_DIM + 2 * HAND_DIM  # 258

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
BLANK_TOKEN = "<blank>"


def build_gloss_vocab(gloss_sentences: list[str], min_freq: int = 1) -> dict[str, int]:
    from collections import Counter
    counter: Counter[str] = Counter()
    for g in gloss_sentences:
        for t in g.split("/"):
            t = t.strip()
            if t:
                counter[t] += 1
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3, BLANK_TOKEN: 4}
    idx = 5
    for tok in sorted(counter):
        if counter[tok] >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab


class KeypointDataset(Dataset):
    """Load pre-extracted MediaPipe keypoints for CSLR."""

    def __init__(
        self,
        split: str,
        vocab: dict[str, int] | None = None,
        min_freq: int = 1,
        augment: bool = False,
        num_frames: int | None = NUM_FRAMES,
    ):
        self.split = split
        self.augment = augment
        self.num_frames = num_frames

        label_path = DATA_DIR / "label" / f"{split}.csv"
        self.samples: list[dict] = []
        with open(label_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                kp_path = KEYPOINT_DIR / split / row["Translator"] / f"{row['Number']}.npy"
                if kp_path.exists():
                    self.samples.append({
                        "id": row["Number"],
                        "translator": row["Translator"],
                        "gloss": gloss,
                        "kp_path": str(kp_path),
                    })

        self.vocab = vocab
        if self.vocab is None:
            all_gloss = [s["gloss"] for s in self.samples]
            self.vocab = build_gloss_vocab(all_gloss, min_freq=min_freq)
        self.idx_to_token = {v: k for k, v in self.vocab.items()}

        n = len(self.samples)
        print(f"  [{split}] {n} samples ready (vocab={self.vocab_size})")

    def __len__(self) -> int:
        return len(self.samples)

    # ---- keypoint loading ---------------------------------------------------

    @staticmethod
    def _load_keypoints(kp_path: str) -> np.ndarray:
        """Load a .npy keypoint file, flatten to [T, 258] float32."""
        data = np.load(kp_path, allow_pickle=True).item()
        pose = data["pose"].reshape(data["pose"].shape[0], -1)              # (T, 132)
        lh = data["left_hand"].reshape(data["left_hand"].shape[0], -1)     # (T, 63)
        rh = data["right_hand"].reshape(data["right_hand"].shape[0], -1)   # (T, 63)
        return np.concatenate([pose, lh, rh], axis=1).astype(np.float32)

    # ---- contiguous sampling -----------------------------------------------

    def _sample_contiguous(self, kp: np.ndarray) -> tuple[np.ndarray, int]:
        T = kp.shape[0]
        if self.num_frames is None:
            return kp, T
        if T >= self.num_frames:
            start = random.randint(0, T - self.num_frames) if self.augment else 0
            return kp[start:start + self.num_frames], self.num_frames
        pad = np.repeat(kp[-1:], self.num_frames - T, axis=0)
        return np.concatenate([kp, pad], axis=0), T

    # ---- augmentations -----------------------------------------------------

    def _augment(self, kp: np.ndarray) -> np.ndarray:
        """Apply a random subset of keypoint-specific augmentations."""
        if random.random() < 0.5:
            kp = self._mirror(kp)
        if random.random() < 0.4:
            kp = self._speed_perturb(kp)
        if random.random() < 0.8:
            kp = self._spatial_jitter(kp)
        if random.random() < 0.6:
            kp = self._keypoint_dropout(kp)
        if random.random() < 0.4:
            kp = self._scale_perturb(kp)
        return kp

    @staticmethod
    def _mirror(kp: np.ndarray) -> np.ndarray:
        """Horizontal flip: invert x coords, swap left/right hands."""
        kp = kp.copy()
        # Invert x for every landmark: each has 3 coords (x,y,z) or 4 (x,y,z,vis)
        for offset, dim, stride in [(0, POSE_DIM, 4), (POSE_DIM, HAND_DIM, 3),
                                      (POSE_DIM + HAND_DIM, HAND_DIM, 3)]:
            present = KeypointDataset._landmark_presence(kp, offset, dim, stride)
            for i in range(dim // stride):
                x_idx = offset + i * stride
                if x_idx < kp.shape[1]:
                    kp[present[:, i], x_idx] = 1.0 - kp[present[:, i], x_idx]
        # Swap left and right hand blocks
        hs = HAND_DIM
        ls = POSE_DIM
        rs = POSE_DIM + HAND_DIM
        lh_bak = kp[:, ls:ls + hs].copy()
        kp[:, ls:ls + hs] = kp[:, rs:rs + hs]
        kp[:, rs:rs + hs] = lh_bak
        return kp

    @staticmethod
    def _speed_perturb(kp: np.ndarray) -> np.ndarray:
        """Linear interpolation to 0.8×–1.2× speed."""
        T = kp.shape[0]
        factor = random.uniform(0.8, 1.2)
        new_T = max(1, int(T * factor))
        new_idx = np.linspace(0, T - 1, new_T)
        out = np.zeros((new_T, kp.shape[1]), dtype=kp.dtype)
        for c in range(kp.shape[1]):
            out[:, c] = np.interp(new_idx, np.arange(T), kp[:, c])
        return out

    @staticmethod
    def _spatial_jitter(kp: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*kp.shape).astype(np.float32) * 0.005
        mask = np.zeros_like(kp, dtype=np.float32)
        for offset, dim, stride in [(0, POSE_DIM, 4), (POSE_DIM, HAND_DIM, 3),
                                      (POSE_DIM + HAND_DIM, HAND_DIM, 3)]:
            present = KeypointDataset._landmark_presence(kp, offset, dim, stride)
            for i in range(dim // stride):
                start = offset + i * stride
                coord_dim = 3  # x/y/z only; pose visibility is not jittered
                mask[present[:, i], start:start + coord_dim] = 1.0
        return kp + noise * mask

    @staticmethod
    def _keypoint_dropout(kp: np.ndarray) -> np.ndarray:
        """Zero out 10% of coordinate values randomly."""
        mask = np.random.rand(*kp.shape) >= 0.1
        return kp * mask.astype(np.float32)

    @staticmethod
    def _temporal_crop(kp: np.ndarray) -> np.ndarray:
        """Randomly crop to 70%-100% of original length."""
        T = kp.shape[0]
        crop_ratio = random.uniform(0.7, 1.0)
        crop_len = max(1, int(T * crop_ratio))
        start = random.randint(0, T - crop_len)
        return kp[start:start + crop_len]

    @staticmethod
    def _scale_perturb(kp: np.ndarray) -> np.ndarray:
        """Scale x,y coordinates by 0.85–1.15."""
        kp = kp.copy()
        factor = random.uniform(0.85, 1.15)
        for offset, dim, stride in [(0, POSE_DIM, 4), (POSE_DIM, HAND_DIM, 3),
                                      (POSE_DIM + HAND_DIM, HAND_DIM, 3)]:
            present = KeypointDataset._landmark_presence(kp, offset, dim, stride)
            for i in range(dim // stride):
                x_idx = offset + i * stride
                y_idx = x_idx + 1
                if x_idx < kp.shape[1]:
                    kp[present[:, i], x_idx] *= factor
                if y_idx < kp.shape[1]:
                    kp[present[:, i], y_idx] *= factor
        return kp

    @staticmethod
    def _landmark_presence(kp: np.ndarray, offset: int, dim: int, stride: int) -> np.ndarray:
        """Return [T, N] mask for landmarks that are not the all-zero sentinel."""
        block = kp[:, offset:offset + dim].reshape(kp.shape[0], dim // stride, stride)
        return np.any(block != 0.0, axis=2)

    # ---- main --------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        kp = self._load_keypoints(sample["kp_path"])  # (T_full, 258)

        if self.augment:
            kp = self._augment(kp)

        kp, valid_len = self._sample_contiguous(kp)  # (T, 258)

        gloss_tokens = [t.strip() for t in sample["gloss"].split("/") if t.strip()]
        token_ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in gloss_tokens]

        valid_mask = torch.zeros(kp.shape[0], dtype=torch.bool)
        valid_mask[:valid_len] = True

        return {
            "keypoints": torch.from_numpy(kp),          # (T, 258)
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "valid_mask": valid_mask,
        }

    # ---- properties --------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def blank_idx(self) -> int:
        return self.vocab[BLANK_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.vocab[UNK_TOKEN]


def collate_keypoint_ctc(batch: list[dict]) -> dict:
    """Collate keypoint sequences into a CTC-compatible batch."""
    max_len = max(b["keypoints"].shape[0] for b in batch)
    kp_dim = batch[0]["keypoints"].shape[1]
    keypoints = torch.zeros(len(batch), max_len, kp_dim, dtype=batch[0]["keypoints"].dtype)
    valid_masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        length = item["keypoints"].shape[0]
        keypoints[i, :length] = item["keypoints"]
        valid_masks[i, :length] = item["valid_mask"]
    input_lengths = valid_masks.sum(dim=1).long()
    tokens_list = [b["tokens"] for b in batch]
    target_lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)
    tokens_cat = torch.cat(tokens_list)

    return {
        "keypoints": keypoints,           # (B, T, 258)
        "tokens": tokens_cat,             # (sum(target_lengths),)
        "input_lengths": input_lengths,   # (B,)
        "target_lengths": target_lengths, # (B,)
        "valid_mask": valid_masks,        # (B, T)
    }
