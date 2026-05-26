"""End-to-end video dataset: loads raw frames for ResNet18 processing on-the-fly."""
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = Path(os.environ.get("CE_CSL_ROOT", "data/CE-CSL"))
VIDEO_DIR = DATA_DIR / "video"
SHOUYU_ROOT = Path(__file__).resolve().parent
FRAME_CACHE_DIR = SHOUYU_ROOT / "e2e_frame_cache"

NUM_FRAMES = 64
FRAME_SIZE = 256  # resize to this first, then random crop to 224
CROP_SIZE = 224

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
BLANK_TOKEN = "<blank>"


def build_gloss_vocab(gloss_sentences: list[str], min_freq: int = 1) -> dict[str, int]:
    from collections import Counter
    counter = Counter()
    for g in gloss_sentences:
        for t in g.split("/"):
            t = t.strip()
            if t:
                counter[t] += 1
    vocab = {PAD_TOKEN: 0, BOS_TOKEN: 1, EOS_TOKEN: 2, UNK_TOKEN: 3, BLANK_TOKEN: 4}
    idx = 5
    for tok in sorted(counter.keys()):
        if counter[tok] >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab


class E2EVideoDataset(Dataset):
    """Load video frames on-the-fly for end-to-end ResNet18 training."""

    def __init__(
        self,
        split: str,
        vocab: dict[str, int] | None = None,
        min_freq: int = 3,
        augment: bool = False,
        use_frame_cache: bool = False,
        frame_cache_dir: str | Path | None = None,
    ):
        self.split = split
        self.augment = augment
        self.use_frame_cache = use_frame_cache
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir is not None else FRAME_CACHE_DIR

        label_path = DATA_DIR / "label" / f"{split}.csv"
        self.samples = []
        with open(label_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gloss = row["Gloss"].strip()
                if not gloss:
                    continue
                video_path = VIDEO_DIR / split / row["Translator"] / f"{row['Number']}.mp4"
                if video_path.exists():
                    cache_path = self.frame_cache_dir / split / row["Translator"] / f"{row['Number']}.npy"
                    self.samples.append({
                        "id": row["Number"],
                        "translator": row["Translator"],
                        "gloss": gloss,
                        "video_path": str(video_path),
                        "cache_path": str(cache_path),
                    })

        self.vocab = vocab
        if self.vocab is None:
            all_gloss = [s["gloss"] for s in self.samples]
            self.vocab = build_gloss_vocab(all_gloss, min_freq=min_freq)

        self.idx_to_token = {v: k for k, v in self.vocab.items()}

        print(f"  [{split}] {len(self.samples)} samples ready")
        if self.use_frame_cache:
            print(f"  [{split}] frame cache: {self.frame_cache_dir}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
        if total_frames <= 0:
            return []
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    @staticmethod
    def _load_frames(video_path: str, num_frames: int) -> np.ndarray:
        """Sample num_frames evenly from video. Returns [T, H, W, C] float32 (0-1)."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return np.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        indices = E2EVideoDataset._sample_frame_indices(total_frames, num_frames)

        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
            frames.append(frame)

        cap.release()

        if not frames:
            return np.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.float32)

        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

        return np.stack(frames, axis=0).astype(np.float32) / 255.0

    @staticmethod
    def _encode_cached_frames(frames: np.ndarray) -> np.ndarray:
        """Store resized RGB frames compactly as uint8 [T, H, W, C]."""
        if frames.dtype == np.uint8:
            return frames
        frames = np.clip(frames, 0.0, 1.0)
        return (frames * 255.0).round().astype(np.uint8)

    @staticmethod
    def _decode_cached_frames(frames: np.ndarray, num_frames: int) -> np.ndarray:
        """Load cached uint8 frames as float32 in 0-1 range with stable length."""
        if frames.shape[0] > num_frames:
            frames = frames[:num_frames]
        if frames.shape[0] < num_frames:
            if frames.shape[0] == 0:
                frames = np.zeros((num_frames, FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
            else:
                pad = np.repeat(frames[-1:],
                                repeats=num_frames - frames.shape[0],
                                axis=0)
                frames = np.concatenate([frames, pad], axis=0)
        if frames.dtype == np.uint8:
            return frames.astype(np.float32) / 255.0
        return frames.astype(np.float32)

    @classmethod
    def _load_frames_cached(cls, video_path: str, cache_path: str, num_frames: int) -> np.ndarray:
        cache = Path(cache_path)
        if cache.exists():
            try:
                cached = np.load(str(cache), allow_pickle=False)
                return cls._decode_cached_frames(cached, num_frames)
            except (OSError, ValueError):
                pass

        frames = cls._load_frames(video_path, num_frames)
        encoded = cls._encode_cached_frames(frames)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npy")
        try:
            np.save(str(tmp_path), encoded)
            tmp_path.replace(cache)
        except OSError:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return frames

    @staticmethod
    def _spatial_augment(frames: torch.Tensor) -> torch.Tensor:
        """Spatial augmentation on [T, C, H, W] tensor (0-1 range)."""
        T, C, H, W = frames.shape
        # Random crop (H=256 -> 224)
        if H > CROP_SIZE:
            top = random.randint(0, H - CROP_SIZE)
            left = random.randint(0, W - CROP_SIZE)
            frames = frames[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]
        # Horizontal flip
        if random.random() < 0.5:
            frames = frames.flip(-1)
        # Color jitter
        brightness = 0.8 + random.random() * 0.4
        contrast = 0.8 + random.random() * 0.4
        saturation = 0.8 + random.random() * 0.4
        for i in range(T):
            frame = frames[i]
            frame = frame * brightness
            mean = frame.mean(dim=(1, 2), keepdim=True)
            frame = (frame - mean) * contrast + mean
            gray = frame.mean(dim=0, keepdim=True)
            frame = gray * (1 - saturation) + frame * saturation
            frame = torch.clamp(frame, 0.0, 1.0)
            frames[i] = frame
        return frames

    @staticmethod
    def _temporal_augment(frames: torch.Tensor) -> torch.Tensor:
        """Temporal augmentation on [T, C, H, W] tensor."""
        T = frames.shape[0]
        # Frame dropout
        if random.random() < 0.5 and T > 2:
            n_drop = random.randint(1, max(1, int(T * 0.15)))
            drop_idx = random.sample(range(T), n_drop)
            frames[drop_idx] = 0.0
        # Block masking
        if T > 8 and random.random() < 0.3:
            mask_len = random.randint(2, max(2, T // 8))
            mask_start = random.randint(0, T - mask_len)
            frames[mask_start:mask_start + mask_len] = 0.0
        return frames

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_frame_cache:
            frames = self._load_frames_cached(sample["video_path"], sample["cache_path"], NUM_FRAMES)
        else:
            frames = self._load_frames(sample["video_path"], NUM_FRAMES)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

        if self.augment:
            frames = self._spatial_augment(frames)
            frames = self._temporal_augment(frames)
        else:
            # Center crop for eval
            H, W = frames.shape[2], frames.shape[3]
            top = (H - CROP_SIZE) // 2
            left = (W - CROP_SIZE) // 2
            frames = frames[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]

        # ImageNet normalize
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD

        # Tokenize gloss
        gloss_tokens = [t.strip() for t in sample["gloss"].split("/") if t.strip()]
        token_ids = [self.vocab.get(tok, self.vocab[UNK_TOKEN]) for tok in gloss_tokens]

        valid_mask = torch.ones(NUM_FRAMES, dtype=torch.bool)

        return {
            "frames": frames,  # [T, C, H, W]
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "valid_mask": valid_mask,
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def blank_idx(self):
        return self.vocab[BLANK_TOKEN]

    @property
    def unk_idx(self):
        return self.vocab[UNK_TOKEN]


def collate_e2e_ctc(batch: list[dict]) -> dict:
    max_T = max(b["frames"].shape[0] for b in batch)
    _, C, H, W = batch[0]["frames"].shape
    B = len(batch)

    frames_padded = torch.zeros(B, max_T, C, H, W)
    valid_masks = torch.zeros(B, max_T, dtype=torch.bool)
    for i, b in enumerate(batch):
        T = b["frames"].shape[0]
        frames_padded[i, :T] = b["frames"]
        valid_masks[i, :T] = b["valid_mask"]

    input_lengths = valid_masks.sum(dim=1).long()
    tokens_list = [b["tokens"] for b in batch]
    target_lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)
    tokens_cat = torch.cat(tokens_list)

    return {
        "frames": frames_padded,
        "tokens": tokens_cat,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "valid_mask": valid_masks,
    }
