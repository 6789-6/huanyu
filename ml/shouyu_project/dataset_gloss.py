"""Dataset for Sign2Gloss and Gloss2Text tasks."""
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

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def build_char_vocab(sentences: list[str]) -> dict[str, int]:
    chars = set()
    for s in sentences:
        chars.update(s)
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    for i, ch in enumerate(sorted(chars), start=4):
        vocab[ch] = i
    return vocab


def build_gloss_vocab(gloss_sentences: list[str], min_freq: int = 1) -> dict[str, int]:
    from collections import Counter
    counter = Counter()
    for g in gloss_sentences:
        for t in g.split("/"):
            t = t.strip()
            if t:
                counter[t] += 1
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3}
    idx = 4
    for tok in sorted(counter.keys()):
        if counter[tok] >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab


def normalize_keypoints(pose, left_hand, right_hand):
    """Normalize pose to body-center and hands to wrist-center."""
    T = pose.shape[0]

    # Body center: midpoint of shoulders (landmarks 11, 12)
    left_shoulder = pose[:, 11, :3]
    right_shoulder = pose[:, 12, :3]
    body_center = (left_shoulder + right_shoulder) / 2.0

    # Body scale: shoulder-hip distance
    left_hip = pose[:, 23, :3]
    right_hip = pose[:, 24, :3]
    hip_center = (left_hip + right_hip) / 2.0
    body_scale = np.linalg.norm(body_center - hip_center, axis=-1, keepdims=True)
    body_scale = np.maximum(body_scale, 1e-6)

    pose_norm = pose.copy()
    # Only normalize if shoulders are detected (non-zero)
    valid = (body_scale.squeeze(-1) > 1e-5)
    if valid.any():
        pose_norm[valid, :, :3] = (
            (pose[valid, :, :3] - body_center[valid, None, :])
            / body_scale[valid, None, :]
        )

    # Hands: normalize relative to wrist (landmark 0)
    left_norm = left_hand.copy()
    if T > 0 and left_hand.shape[0] == T:
        left_wrist = left_hand[:, 0, :3]
        left_tip = left_hand[:, 12, :3]
        left_scale = np.linalg.norm(left_tip - left_wrist, axis=-1, keepdims=True)
        left_scale = np.maximum(left_scale, 1e-6)
        valid_l = (left_scale.squeeze(-1) > 1e-5)
        if valid_l.any():
            left_norm[valid_l, :, :3] = (
                (left_hand[valid_l, :, :3] - left_wrist[valid_l, None, :])
                / left_scale[valid_l, None, :]
            )

    right_norm = right_hand.copy()
    if T > 0 and right_hand.shape[0] == T:
        right_wrist = right_hand[:, 0, :3]
        right_tip = right_hand[:, 12, :3]
        right_scale = np.linalg.norm(right_tip - right_wrist, axis=-1, keepdims=True)
        right_scale = np.maximum(right_scale, 1e-6)
        valid_r = (right_scale.squeeze(-1) > 1e-5)
        if valid_r.any():
            right_norm[valid_r, :, :3] = (
                (right_hand[valid_r, :, :3] - right_wrist[valid_r, None, :])
                / right_scale[valid_r, None, :]
            )

    return pose_norm, left_norm, right_norm


def add_motion_features(kp: np.ndarray) -> np.ndarray:
    """Concatenate position and first-order frame differences (velocity)."""
    vel = np.zeros_like(kp)
    vel[1:] = kp[1:] - kp[:-1]
    return np.concatenate([kp, vel], axis=-1)


INPUT_DIM = 516  # 258 normalized + 258 velocity


class Sign2GlossDataset(Dataset):
    """Keypoints -> Gloss tokens."""

    def __init__(self, split: str, vocab: dict[str, int] | None = None,
                 max_len: int = 150, augment: bool = False, min_freq: int = 1):
        self.split = split
        self.max_len = max_len
        self.augment = augment

        label_path = DATA_DIR / "label" / f"{split}.csv"
        self.samples = []
        with open(label_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                self.samples.append({
                    "id": row["Number"],
                    "translator": row["Translator"],
                    "gloss": gloss,
                })

        self.vocab = vocab
        if self.vocab is None:
            all_gloss = [s["gloss"] for s in self.samples]
            self.vocab = build_gloss_vocab(all_gloss, min_freq=min_freq)

        self.idx_to_token = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.samples)

    def _augment(self, keypoints: np.ndarray, T: int) -> np.ndarray:
        kp = keypoints.copy()
        real_mask = kp[:T] != 0
        noise = np.random.randn(T, kp.shape[1]).astype(np.float32) * 0.005
        kp[:T] = kp[:T] + noise * real_mask
        n_drop = max(1, int(T * 0.15))
        drop_idx = np.random.choice(T, n_drop, replace=False)
        kp[drop_idx] = 0
        if T > 8:
            mask_len = random.randint(2, max(2, T // 8))
            mask_start = random.randint(0, T - mask_len)
            kp[mask_start:mask_start + mask_len] = 0
        return kp

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample["id"]
        translator = sample["translator"]
        npy_path = KEYPOINT_DIR / self.split / translator / f"{sample_id}.npy"

        data = np.load(str(npy_path), allow_pickle=True).item()
        T = data["num_frames"]
        pose = data["pose"]        # [T, 33, 4]
        left = data["left_hand"]   # [T, 21, 3]
        right = data["right_hand"] # [T, 21, 3]

        # Normalize: body-centric for pose, wrist-centric for hands
        pose, left, right = normalize_keypoints(pose, left, right)

        # Flatten and concatenate
        keypoints = np.concatenate([
            pose.reshape(T, -1),     # 132
            left.reshape(T, -1),     # 63
            right.reshape(T, -1),    # 63
        ], axis=-1)  # [T, 258]

        # Add velocity
        keypoints = add_motion_features(keypoints)  # [T, 516]

        # Pad or truncate
        if T > self.max_len:
            keypoints = keypoints[:self.max_len]
            T = self.max_len
        else:
            pad = np.zeros((self.max_len - T, keypoints.shape[1]), dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)

        if self.augment:
            keypoints = self._augment(keypoints, T)

        # Tokenize gloss
        gloss = sample["gloss"]
        gloss_tokens = [t.strip() for t in gloss.split("/") if t.strip()]
        token_ids = [self.vocab[BOS_TOKEN]]
        for tok in gloss_tokens:
            token_ids.append(self.vocab.get(tok, self.vocab[UNK_TOKEN]))
        token_ids.append(self.vocab[EOS_TOKEN])

        return {
            "keypoints": torch.from_numpy(keypoints).float(),
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "length": T,
            "text": gloss,
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_idx(self):
        return self.vocab[PAD_TOKEN]

    @property
    def bos_idx(self):
        return self.vocab[BOS_TOKEN]

    @property
    def eos_idx(self):
        return self.vocab[EOS_TOKEN]


class Gloss2TextDataset(Dataset):
    """Gloss tokens -> Chinese characters."""

    def __init__(self, split: str, gloss_vocab: dict[str, int],
                 char_vocab: dict[str, int] | None = None, max_gloss_len: int = 32):
        self.split = split
        self.max_gloss_len = max_gloss_len
        self.gloss_vocab = gloss_vocab

        label_path = DATA_DIR / "label" / f"{split}.csv"
        self.samples = []
        with open(label_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                self.samples.append({
                    "gloss": gloss,
                    "chinese": row["Chinese Sentences"],
                })

        self.char_vocab = char_vocab
        if self.char_vocab is None:
            all_chinese = [s["chinese"] for s in self.samples]
            self.char_vocab = build_char_vocab(all_chinese)

        self.idx_to_char = {v: k for k, v in self.char_vocab.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Source: gloss tokens
        gloss = sample["gloss"]
        gloss_tokens = [t.strip() for t in gloss.split("/") if t.strip()]
        src_ids = [self.gloss_vocab[BOS_TOKEN]]
        for tok in gloss_tokens:
            src_ids.append(self.gloss_vocab.get(tok, self.gloss_vocab[UNK_TOKEN]))
        src_ids.append(self.gloss_vocab[EOS_TOKEN])

        # Pad gloss to max_gloss_len
        if len(src_ids) > self.max_gloss_len:
            src_ids = src_ids[:self.max_gloss_len]
            src_len = self.max_gloss_len
        else:
            src_len = len(src_ids)
            src_ids = src_ids + [self.gloss_vocab[PAD_TOKEN]] * (self.max_gloss_len - len(src_ids))

        # Target: Chinese characters
        chinese = sample["chinese"]
        tgt_ids = [self.char_vocab[BOS_TOKEN]]
        for ch in chinese:
            tgt_ids.append(self.char_vocab.get(ch, self.char_vocab[UNK_TOKEN]))
        tgt_ids.append(self.char_vocab[EOS_TOKEN])

        return {
            "src_tokens": torch.tensor(src_ids, dtype=torch.long),
            "tgt_tokens": torch.tensor(tgt_ids, dtype=torch.long),
            "src_length": src_len,
        }

    @property
    def src_vocab_size(self):
        return len(self.gloss_vocab)

    @property
    def tgt_vocab_size(self):
        return len(self.char_vocab)

    @property
    def src_pad_idx(self):
        return self.gloss_vocab[PAD_TOKEN]

    @property
    def tgt_pad_idx(self):
        return self.char_vocab[PAD_TOKEN]


def collate_gloss(batch: list[dict]) -> dict:
    """Collate for Sign2Gloss (same structure as original)."""
    keypoints = torch.stack([b["keypoints"] for b in batch])
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)

    tokens = [b["tokens"] for b in batch]
    max_tok_len = max(len(t) for t in tokens)
    padded_tokens = torch.full((len(batch), max_tok_len), 0, dtype=torch.long)
    for i, t in enumerate(tokens):
        padded_tokens[i, :len(t)] = t

    src_mask = torch.arange(keypoints.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    tgt_mask = padded_tokens != 0

    return {
        "keypoints": keypoints,
        "tokens": padded_tokens,
        "lengths": lengths,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }


def collate_gloss2text(batch: list[dict]) -> dict:
    """Collate for Gloss2Text: gloss sequence -> Chinese characters."""
    src_tokens = torch.stack([b["src_tokens"] for b in batch])
    src_lengths = torch.tensor([b["src_length"] for b in batch], dtype=torch.long)

    tgt = [b["tgt_tokens"] for b in batch]
    max_tgt_len = max(len(t) for t in tgt)
    padded_tgt = torch.full((len(batch), max_tgt_len), 0, dtype=torch.long)
    for i, t in enumerate(tgt):
        padded_tgt[i, :len(t)] = t

    src_mask = torch.arange(src_tokens.shape[1]).unsqueeze(0) < src_lengths.unsqueeze(1)
    tgt_mask = padded_tgt != 0

    return {
        "src_tokens": src_tokens,
        "tgt_tokens": padded_tgt,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
    }
