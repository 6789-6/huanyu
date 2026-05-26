#!/usr/bin/env python3
# ========================================================
# 唤语 - 手语识别数据集加载器
# 支持 CSL (Chinese Sign Language) 数据集
# ========================================================

import os
import sys
import json
from pathlib import Path
from typing import List, Tuple, Optional, Callable

# 延迟导入重型依赖
try:
    import cv2
    import numpy as np
    HAS_CV = True
except ImportError:
    HAS_CV = False
    print("Warning: opencv-python not installed")

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed")
    # 定义占位类用于测试
    class Dataset:
        pass
    class DataLoader:
        def __init__(self, *args, **kwargs):
            pass


class CSLDataset(Dataset):
    """
    CSL (Chinese Sign Language) 数据集加载器
    
    数据集结构:
    CSL/
    ├── train/
    │   ├── word_001/
    │   │   ├── video_001.mp4
    │   │   ├── video_002.mp4
    │   │   └── ...
    │   ├── word_002/
    │   └── ...
    ├── test/
    └── vocab.json
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train' 或 'test'
            num_frames: 每个视频采样的帧数
            frame_size: 帧大小 (H, W)
            transform: 数据增强变换
            cache_dir: 预处理缓存目录
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # 加载词汇表
        self.vocab_path = self.data_dir / 'vocab.json'
        self.word_to_idx, self.idx_to_word = self._load_vocab()
        self.num_classes = len(self.word_to_idx)
        
        # 构建样本列表
        self.samples = self._build_samples()
        print(f"CSL {split} dataset: {len(self.samples)} samples, {self.num_classes} classes")
    
    def _load_vocab(self) -> Tuple[dict, dict]:
        """加载词汇表"""
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            word_to_idx = {word: idx for idx, word in enumerate(vocab)}
            idx_to_word = {idx: word for idx, word in enumerate(vocab)}
        else:
            # 如果没有词汇表文件，从目录结构推断
            word_to_idx = {}
            idx_to_word = {}
            split_dir = self.data_dir / self.split
            if split_dir.exists():
                for idx, word_dir in enumerate(sorted(split_dir.iterdir())):
                    if word_dir.is_dir():
                        word = word_dir.name
                        word_to_idx[word] = idx
                        idx_to_word[idx] = word
            
            # 保存词汇表
            if word_to_idx:
                vocab = [idx_to_word[i] for i in range(len(idx_to_word))]
                with open(self.vocab_path, 'w', encoding='utf-8') as f:
                    json.dump(vocab, f, ensure_ascii=False, indent=2)
        
        return word_to_idx, idx_to_word
    
    def _build_samples(self) -> List[Tuple[str, int]]:
        """构建样本列表 (video_path, label)"""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            return samples
        
        for word_dir in split_dir.iterdir():
            if not word_dir.is_dir():
                continue
            
            word = word_dir.name
            if word not in self.word_to_idx:
                continue
            
            label = self.word_to_idx[word]
            
            # 查找该词的所有视频
            for video_path in word_dir.glob('*.mp4'):
                samples.append((str(video_path), label))
            for video_path in word_dir.glob('*.avi'):
                samples.append((str(video_path), label))
        
        return samples
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """
        加载视频并采样帧
        
        Returns:
            frames: (T, H, W, C) numpy array, uint8
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.uint8)
        
        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.uint8)
        
        # 均匀采样帧索引
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # 如果视频帧数不足，重复最后一帧
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(self.num_frames - total_frames, total_frames - 1)
            ])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                # 如果读取失败，用前一帧或黑帧
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((*self.frame_size, 3), dtype=np.uint8)
            else:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            
            frames.append(frame)
        
        cap.release()
        
        return np.array(frames, dtype=np.uint8)
    
    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """
        预处理帧
        
        Args:
            frames: (T, H, W, C) uint8 [0, 255]
        
        Returns:
            tensor: (C, T, H, W) float32 [0, 1]
        """
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0
        
        # (T, H, W, C) -> (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        # 标准化 (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1, 1)
        frames = (frames - mean) / std
        
        return torch.from_numpy(frames).float()
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        
        # 加载视频
        frames = self._load_video(video_path)
        
        # 预处理
        video_tensor = self._preprocess_frames(frames)
        
        if self.transform:
            video_tensor = self.transform(video_tensor)
        
        return video_tensor, label


class SyntheticCSLDataset(Dataset):
    """
    合成数据集 - 用于快速验证训练流程
    在没有真实数据集时使用
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 100,
        num_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # 生成词汇表
        self.idx_to_word = {i: f"word_{i:03d}" for i in range(num_classes)}
        self.word_to_idx = {v: k for k, v in self.idx_to_word.items()}
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # 生成随机视频数据 (模拟手语动作)
        # 使用正弦波模拟时序特征
        label = idx % self.num_classes
        
        # 基础模式 + 随机噪声
        t = np.linspace(0, 2 * np.pi * (label + 1) / self.num_classes, self.num_frames)
        
        # 生成时空特征
        frames = np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.float32)
        
        for i in range(self.num_frames):
            # 基于标签生成不同的模式
            phase = t[i]
            intensity = (np.sin(phase) + 1) / 2  # [0, 1]
            
            # 在图像中心生成移动的"手"
            center_x = self.frame_size[1] // 2 + int(50 * np.cos(phase * 2))
            center_y = self.frame_size[0] // 2 + int(30 * np.sin(phase * 3))
            
            # 绘制圆形
            y, x = np.ogrid[:self.frame_size[0], :self.frame_size[1]]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < (30 + 10 * intensity) ** 2
            
            # 颜色随标签变化
            color = np.array([
                0.5 + 0.5 * np.sin(phase),
                0.5 + 0.5 * np.cos(phase),
                0.5 + 0.5 * np.sin(phase * 2)
            ])
            
            frames[i, mask] = color
        
        # 添加噪声
        frames += np.random.randn(*frames.shape) * 0.1
        frames = np.clip(frames, 0, 1)
        
        # (T, H, W, C) -> (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))
        
        # 标准化
        mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1, 1)
        frames = (frames - mean) / std
        
        return torch.from_numpy(frames).float(), label


def get_data_loaders(
    data_dir: str,
    batch_size: int = 4,
    num_frames: int = 32,
    frame_size: Tuple[int, int] = (224, 224),
    num_workers: int = 2,
    use_synthetic: bool = False
) -> Tuple[DataLoader, DataLoader, int]:
    """
    获取数据加载器
    
    Returns:
        train_loader, test_loader, num_classes
    """
    if use_synthetic:
        print("Using synthetic dataset for testing...")
        train_dataset = SyntheticCSLDataset(
            num_samples=1000,
            num_classes=100,
            num_frames=num_frames,
            frame_size=frame_size
        )
        test_dataset = SyntheticCSLDataset(
            num_samples=200,
            num_classes=100,
            num_frames=num_frames,
            frame_size=frame_size
        )
    else:
        train_dataset = CSLDataset(
            data_dir=data_dir,
            split='train',
            num_frames=num_frames,
            frame_size=frame_size
        )
        test_dataset = CSLDataset(
            data_dir=data_dir,
            split='test',
            num_frames=num_frames,
            frame_size=frame_size
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.num_classes


# ========================================================
# 测试代码
# ========================================================

if __name__ == '__main__':
    import time
    
    print("=" * 50)
    print("唤语 - 数据集加载器测试")
    print("=" * 50)
    print(f"OpenCV available: {HAS_CV}")
    print(f"PyTorch available: {HAS_TORCH}")
    print("=" * 50)
    
    if not HAS_TORCH:
        print("\n⚠️  PyTorch 未安装，跳过完整测试")
        print("\n安装 PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("\n测试完成！")
        sys.exit(0)
    
    # 测试合成数据集
    print("\nTesting Synthetic Dataset...")
    dataset = SyntheticCSLDataset(num_samples=10, num_classes=5)
    print(f"Dataset size: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")
    
    video, label = dataset[0]
    print(f"Video shape: {video.shape}")  # (C, T, H, W)
    print(f"Label: {label}")
    
    # 测试 DataLoader
    train_loader, test_loader, num_classes = get_data_loaders(
        data_dir='',
        batch_size=2,
        use_synthetic=True
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Num classes: {num_classes}")
    
    # 测试加载速度
    start = time.time()
    for batch_idx, (videos, labels) in enumerate(train_loader):
        if batch_idx >= 5:
            break
        print(f"Batch {batch_idx}: videos {videos.shape}, labels {labels.shape}")
    elapsed = time.time() - start
    print(f"\nLoading 5 batches took {elapsed:.2f}s")
    
    print("\n" + "=" * 50)
    print("✅ Dataset loader test completed!")
    print("=" * 50)