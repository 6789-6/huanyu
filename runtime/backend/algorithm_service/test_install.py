#!/usr/bin/env python3
# ========================================================
# 唤语 - 依赖安装测试脚本
# ========================================================

import sys
import subprocess

print("=" * 60)
print("唤语 - 依赖安装测试")
print("=" * 60)

# 检查各项依赖
checks = {
    'Python': lambda: sys.version.split()[0],
    'NumPy': lambda: subprocess.check_output([sys.executable, '-c', 'import numpy; print(numpy.__version__)'], text=True).strip(),
    'OpenCV': lambda: subprocess.check_output([sys.executable, '-c', 'import cv2; print(cv2.__version__)'], text=True).strip(),
    'PyTorch': lambda: subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.__version__)'], text=True).strip(),
}

results = {}
for name, check_fn in checks.items():
    try:
        version = check_fn()
        results[name] = f"✅ {version}"
    except Exception as e:
        results[name] = "❌ 未安装"

print("\n依赖状态:")
print("-" * 60)
for name, status in results.items():
    print(f"  {name:12} {status}")

print("-" * 60)

# 给出建议
missing = [name for name, status in results.items() if "未安装" in status]

if not missing:
    print("\n✅ 所有依赖已安装！可以开始训练了。")
    print("\n运行训练:")
    print("  python3 train.py --use-synthetic --epochs 10")
else:
    print(f"\n⚠️  缺少依赖：{', '.join(missing)}")
    print("\n安装命令:")
    if 'PyTorch' in missing:
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
    if 'OpenCV' in missing or 'NumPy' in missing:
        print("  pip install opencv-python numpy")

print("=" * 60)
