#!/usr/bin/env python3
# ========================================================
# 唤语 - CSL 数据集下载脚本
# ========================================================

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_info():
    """打印数据集信息"""
    print("=" * 60)
    print("CSL (Chinese Sign Language) 数据集")
    print("=" * 60)
    print("""
来源: https://github.com/ustc-slr/SLRDataset

数据集特点:
- 中国手语 (Chinese Sign Language)
- 50,000 个视频样本
- 1,000+ 个词汇类别  
- 由中科院自动化研究所发布
- 学术认可度高

下载方式:
1. 访问 GitHub 仓库: https://github.com/ustc-slr/SLRDataset
2. 按照 README 中的指引申请数据集
3. 通常需要发送邮件给作者说明研究用途

备选方案:
- 使用 tensorflow_datasets 自动下载:
  pip install sign-language-datasets tensorflow-datasets
  python -c "import tensorflow_datasets as tfds; tfds.load('csl')"

- 使用合成数据集测试训练流程:
  python train.py --use-synthetic
""")
    print("=" * 60)


def install_dependencies():
    """安装依赖"""
    print("\nInstalling dependencies...")
    
    packages = [
        'tensorflow-datasets',
        'sign-language-datasets',
        'opencv-python',
        'torch',
        'torchvision',
        'tensorboard',
        'numpy'
    ]
    
    for pkg in packages:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])
    
    print("Dependencies installed!")


def try_tfds_download(data_dir: str):
    """尝试使用 TFDS 下载"""
    print(f"\nTrying to download CSL dataset using TensorFlow Datasets...")
    print(f"Data will be saved to: {data_dir}")
    
    try:
        import tensorflow_datasets as tfds
        
        # 加载 CSL 数据集
        dataset = tfds.load(
            'csl',
            data_dir=data_dir,
            download=True
        )
        
        print("Dataset downloaded successfully!")
        print(f"Dataset info:")
        print(dataset)
        
        return True
    except Exception as e:
        print(f"TFDS download failed: {e}")
        print("\nPlease download manually from:")
        print("https://github.com/ustc-slr/SLRDataset")
        return False


def create_synthetic_dataset(data_dir: str, num_classes: int = 100):
    """创建合成数据集用于测试"""
    print(f"\nCreating synthetic dataset with {num_classes} classes...")
    
    data_path = Path(data_dir)
    
    # 创建目录结构
    for split in ['train', 'test']:
        for i in range(num_classes):
            word_dir = data_path / split / f"word_{i:03d}"
            word_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建词汇表
    vocab = [f"word_{i:03d}" for i in range(num_classes)]
    with open(data_path / 'vocab.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Synthetic dataset structure created at: {data_dir}")
    print("Note: This is just the directory structure.")
    print("Use --use-synthetic flag in train.py to generate data on-the-fly.")


def main():
    parser = argparse.ArgumentParser(description='下载 CSL 手语数据集')
    parser.add_argument('--data-dir', type=str, default='./data/CSL',
                        help='数据集保存目录')
    parser.add_argument('--install-deps', action='store_true',
                        help='安装依赖')
    parser.add_argument('--use-tfds', action='store_true',
                        help='尝试使用 TensorFlow Datasets 下载')
    parser.add_argument('--create-synthetic', action='store_true',
                        help='创建合成数据集结构')
    parser.add_argument('--num-classes', type=int, default=100,
                        help='合成数据集类别数')
    
    args = parser.parse_args()
    
    print_info()
    
    if args.install_deps:
        install_dependencies()
    
    if args.use_tfds:
        try_tfds_download(args.data_dir)
    
    if args.create_synthetic:
        create_synthetic_dataset(args.data_dir, args.num_classes)
    
    if not any([args.install_deps, args.use_tfds, args.create_synthetic]):
        print("\nUsage examples:")
        print("  python download_csl.py --install-deps")
        print("  python download_csl.py --use-tfds")
        print("  python download_csl.py --create-synthetic --num-classes 100")
        print("\nFor manual download, visit:")
        print("  https://github.com/ustc-slr/SLRDataset")


if __name__ == '__main__':
    main()
