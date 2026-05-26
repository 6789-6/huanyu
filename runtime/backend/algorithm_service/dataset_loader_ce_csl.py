#!/usr/bin/env python3
# ========================================================
# 唤语 - CE-CSL 数据集加载器
# 适配 CE-CSL 数据集结构
# ========================================================

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional
from tqdm import tqdm


class CECSLDataset(Dataset):
    """
    CE-CSL 手语数据集加载器
    
    数据集结构:
    CE-CSL/
    ├── label/
    │   ├── train.csv
    │   ├── dev.csv
    │   └── test.csv
    └── video/
        ├── train/
        │   ├── A/
        │   ├── B/
        │   └── ...
        ├── dev/
        └── test/
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        num_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        cache_labels: bool = True
    ):
        """
        Args:
            data_dir: 数据集根目录
            split: 'train', 'dev', 或 'test'
            num_frames: 每个视频采样的帧数
            frame_size: 帧大小 (H, W)
            cache_labels: 是否缓存标签
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # 标签和视频路径
        self.label_file = self.data_dir / 'label' / f'{split}.csv'
        self.video_dir = self.data_dir / 'video' / split
        
        # 构建样本列表
        self.samples = self._load_samples()
        self.num_classes = len(self.label_to_idx)
        
        print(f"✅ CE-CSL {split}: {len(self.samples)} videos, {self.num_classes} classes")
    
    def _load_samples(self) -> list:
        """加载样本列表"""
        samples = []
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        if not self.label_file.exists():
            print(f"❌ 标签文件不存在：{self.label_file}")
            return samples
        
        # 读取 CSV 标签文件
        df = pd.read_csv(self.label_file)
        
        # 自动检测列名
        columns = [c.lower() for c in df.columns]
        
        # 查找视频文件名列
        video_col = None
        for name in ['video', 'video_path', 'filepath', 'file', 'path', 'filename']:
            if name in columns:
                video_col = df.columns[columns.index(name)]
                break
        if video_col is None:
            video_col = df.columns[0]  # 默认第一列
        
        # 查找标签列
        label_col = None
        for name in ['label', 'category', 'class', 'tag', 'word']:
            if name in columns:
                label_col = df.columns[columns.index(name)]
                break
        if label_col is None:
            label_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        print(f"   视频列：{video_col}, 标签列：{label_col}")
        
        # 构建标签映射
        unique_labels = df[label_col].unique()
        for idx, label in enumerate(sorted(unique_labels)):
            self.label_to_idx[label] = idx
            self.idx_to_label[idx] = label
        
        # 构建样本列表
        for _, row in df.iterrows():
            video_name = str(row[video_col])
            label = row[label_col]
            
            # 构建视频路径
            # 尝试多种可能的路径格式
            video_path = None
            
            # 格式 1: video/split/label/video_name.mp4
            label_dir = self.video_dir / str(label)
            if label_dir.exists():
                # 直接是视频文件
                if (label_dir / video_name).exists():
                    video_path = label_dir / video_name
                # 带 .mp4 扩展名
                elif (label_dir / f"{video_name}.mp4").exists():
                    video_path = label_dir / f"{video_name}.mp4"
                else:
                    # 查找第一个 mp4 文件
                    mp4_files = list(label_dir.glob('*.mp4'))
                    if mp4_files:
                        video_path = mp4_files[0]
            
            # 格式 2: video/split/video_name.mp4 (标签在文件名中)
            if video_path is None:
                direct_path = self.video_dir / video_name
                if direct_path.exists():
                    video_path = direct_path
                elif (self.video_dir / f"{video_name}.mp4").exists():
                    video_path = self.video_dir / f"{video_name}.mp4"
            
            if video_path and video_path.exists():
                samples.append((str(video_path), self.label_to_idx[label]))
            else:
                print(f"   ⚠️  视频不存在：{video_name} (标签：{label})")
        
        return samples
    
    def _load_video(self, video_path: str) -> np.ndarray:
        """加载视频并采样帧"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 无法打开视频：{video_path}")
            return np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.uint8)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return np.zeros((self.num_frames, *self.frame_size, 3), dtype=np.uint8)
        
        # 均匀采样
        if total_frames >= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.concatenate([
                np.arange(total_frames),
                np.full(self.num_frames - total_frames, total_frames - 1)
            ])
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            else:
                frame = frames[-1].copy() if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8)
            
            frames.append(frame)
        
        cap.release()
        return np.array(frames, dtype=np.uint8)
    
    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """预处理帧"""
        # 归一化到 [0, 1]
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
        frames = self._load_video(video_path)
        video_tensor = self._preprocess_frames(frames)
        return video_tensor, label
    
    def get_class_name(self, idx: int) -> str:
        """获取类别名称"""
        return self.idx_to_label.get(idx, f"Unknown_{idx}")


def get_ce_csl_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_frames: int = 32,
    frame_size: Tuple[int, int] = (224, 224),
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, int]:
    """
    获取 CE-CSL 数据加载器
    
    Returns:
        train_loader, val_loader, num_classes
    """
    train_dataset = CECSLDataset(
        data_dir=data_dir,
        split='train',
        num_frames=num_frames,
        frame_size=frame_size
    )
    
    val_dataset = CECSLDataset(
        data_dir=data_dir,
        split='dev',  # CE-CSL 使用 dev 作为验证集
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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.num_classes


# ========================================================
# 测试代码
# ========================================================

if __name__ == '__main__':
    import time
    
    print("=" * 60)
    print("CE-CSL 数据集加载器测试")
    print("=" * 60)
    
    data_dir = '/home/hao/.openclaw/workspace/唤语/算法服务/data/CE-CSL'
    
    # 测试数据加载
    print("\n加载训练集...")
    train_loader, val_loader, num_classes = get_ce_csl_loaders(
        data_dir=data_dir,
        batch_size=4,
        num_frames=16,
        frame_size=(112, 112),
        num_workers=0
    )
    
    print(f"\n训练集 batches: {len(train_loader)}")
    print(f"验证集 batches: {len(val_loader)}")
    print(f"类别数：{num_classes}")
    
    # 测试加载速度
    print("\n测试数据加载速度...")
    start = time.time()
    for batch_idx, (videos, labels) in enumerate(train_loader):
        if batch_idx >= 3:
            break
        print(f"Batch {batch_idx}: videos {videos.shape}, labels {labels.shape}")
    elapsed = time.time() - start
    print(f"\n加载 3 个 batches 用时：{elapsed:.2f}s")
    
    print("\n✅ 测试完成！")
