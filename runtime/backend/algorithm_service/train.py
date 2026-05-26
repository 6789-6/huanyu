#!/usr/bin/env python3
# ========================================================
# 唤语 - 手语识别模型训练脚本
# ========================================================

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models import SignLanguageRecognitionModel
from dataset_loader import get_data_loaders


class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    epoch: int,
    device: torch.device,
    log_interval: int = 10
) -> Dict[str, float]:
    """训练一个 epoch"""
    
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    start_time = time.time()
    
    for batch_idx, (videos, labels) in enumerate(train_loader):
        # 移动到设备
        videos = videos.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(videos)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # 更新统计
        losses.update(loss.item(), videos.size(0))
        top1.update(acc1.item(), videos.size(0))
        top5.update(acc5.item(), videos.size(0))
        
        # 打印进度
        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            print(
                f'Epoch [{epoch}][{batch_idx}/{len(train_loader)}] '
                f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                f'Acc@1: {top1.val:.2f}% ({top1.avg:.2f}%) '
                f'Acc@5: {top5.val:.2f}% ({top5.avg:.2f}%) '
                f'Time: {elapsed:.1f}s'
            )
    
    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device
) -> Dict[str, float]:
    """验证模型"""
    
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            labels = labels.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            losses.update(loss.item(), videos.size(0))
            top1.update(acc1.item(), videos.size(0))
            top5.update(acc5.item(), videos.size(0))
    
    print(
        f'Validation: '
        f'Loss: {losses.avg:.4f} '
        f'Acc@1: {top1.avg:.2f}% '
        f'Acc@5: {top5.avg:.2f}%'
    )
    
    return {
        'loss': losses.avg,
        'top1_acc': top1.avg,
        'top5_acc': top5.avg
    }


def save_checkpoint(
    state: Dict,
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = 'checkpoint.pth'
):
    """保存检查点"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_path = checkpoint_dir / 'model_best.pth'
        torch.save(state, best_path)
        print(f"Saved best model to {best_path}")


def main():
    parser = argparse.ArgumentParser(description='唤语手语识别模型训练')
    
    # 数据参数
    parser.add_argument('--data-dir', type=str, default='./data/CSL',
                        help='数据集目录')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='使用合成数据集测试')
    parser.add_argument('--num-frames', type=int, default=32,
                        help='视频帧数')
    parser.add_argument('--frame-size', type=int, default=224,
                        help='帧大小')
    
    # 模型参数
    parser.add_argument('--num-classes', type=int, default=1000,
                        help='类别数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout 率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='数据加载线程数')
    
    # 其他参数
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='检查点保存目录')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='日志目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的检查点路径')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅评估模式')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("唤语 - 手语识别模型训练")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Data dir: {args.data_dir}")
    print(f"Use synthetic: {args.use_synthetic}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # 创建目录
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\nLoading data...")
    train_loader, test_loader, num_classes = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size),
        num_workers=args.num_workers,
        use_synthetic=args.use_synthetic
    )
    
    if args.use_synthetic:
        args.num_classes = num_classes
    
    print(f"Num classes: {args.num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 创建模型
    print("\nCreating model...")
    model = SignLanguageRecognitionModel(
        num_classes=args.num_classes,
        num_frames=args.num_frames
    )
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # TensorBoard
    writer = SummaryWriter(log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # 仅评估模式
    if args.eval_only:
        print("\nEvaluating model...")
        val_stats = validate(model, test_loader, criterion, device)
        return
    
    # 训练循环
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 60)
        
        # 训练
        train_stats = train_epoch(
            model, train_loader, criterion, optimizer,
            epoch, device, log_interval=10
        )
        
        # 验证
        val_stats = validate(model, test_loader, criterion, device)
        
        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # 记录到 TensorBoard
        writer.add_scalar('Train/Loss', train_stats['loss'], epoch)
        writer.add_scalar('Train/Top1_Acc', train_stats['top1_acc'], epoch)
        writer.add_scalar('Train/Top5_Acc', train_stats['top5_acc'], epoch)
        writer.add_scalar('Val/Loss', val_stats['loss'], epoch)
        writer.add_scalar('Val/Top1_Acc', val_stats['top1_acc'], epoch)
        writer.add_scalar('Val/Top5_Acc', val_stats['top5_acc'], epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # 保存检查点
        is_best = val_stats['top1_acc'] > best_acc
        best_acc = max(val_stats['top1_acc'], best_acc)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'args': vars(args)
        }, is_best, checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        
        print(f"Best accuracy so far: {best_acc:.2f}%")
    
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()