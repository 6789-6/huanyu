#!/usr/bin/env python3
# ========================================================
# 唤语 - CE-CSL 数据集分析脚本
# ========================================================

import pandas as pd
from pathlib import Path

# 数据集路径
data_dir = Path('/home/hao/.openclaw/workspace/唤语/算法服务/data/CE-CSL')
label_dir = data_dir / 'label'
video_dir = data_dir / 'video'

print("=" * 60)
print("CE-CSL 数据集分析")
print("=" * 60)

# 检查目录
print(f"\n📁 数据目录：{data_dir}")
print(f"   标签目录：{label_dir}")
print(f"   视频目录：{video_dir}")

# 分析标签文件
print("\n" + "=" * 60)
print("标签文件分析")
print("=" * 60)

for split in ['train', 'dev', 'test']:
    csv_path = label_dir / f'{split}.csv'
    if csv_path.exists():
        print(f"\n📄 {split}.csv:")
        try:
            # 尝试读取 CSV
            df = pd.read_csv(csv_path)
            print(f"   行数：{len(df)}")
            print(f"   列名：{list(df.columns)}")
            print(f"\n   前 5 行数据:")
            print(df.head())
            
            # 统计类别
            if 'label' in df.columns or 'Label' in df.columns or 'category' in df.columns:
                label_col = [c for c in df.columns if c.lower() in ['label', 'category', 'class']][0]
                num_classes = df[label_col].nunique()
                print(f"\n   类别数：{num_classes}")
                print(f"   唯一样本：{df[label_col].unique()[:10]}...")
        except Exception as e:
            print(f"   读取失败：{e}")
            # 尝试直接查看文件内容
            with open(csv_path, 'r', encoding='utf-8') as f:
                print("   文件前 5 行:")
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    print(f"   {line.strip()}")
    else:
        print(f"\n❌ {split}.csv 不存在")

# 分析视频目录
print("\n" + "=" * 60)
print("视频目录分析")
print("=" * 60)

for split in ['train', 'dev', 'test']:
    split_dir = video_dir / split
    if split_dir.exists():
        print(f"\n📁 {split}/:")
        categories = [d.name for d in split_dir.iterdir() if d.is_dir()]
        print(f"   类别数：{len(categories)}")
        print(f"   类别列表：{categories[:20]}{'...' if len(categories) > 20 else ''}")
        
        # 统计视频数量
        total_videos = sum(1 for cat in split_dir.iterdir() 
                          if cat.is_dir() 
                          for f in cat.glob('*.mp4'))
        print(f"   视频总数：{total_videos}")
    else:
        print(f"\n❌ {split}/ 不存在")

print("\n" + "=" * 60)
print("分析完成！")
print("=" * 60)
