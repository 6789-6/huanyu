#!/bin/bash
# ========================================================
# 唤语 - 手语识别训练快速启动脚本
# ========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================"
echo "唤语 - 手语识别模型训练"
echo "========================================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python: $PYTHON_VERSION"

# 检查/创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

source venv/bin/activate

# 检查 PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "📦 安装 PyTorch (CPU 版本)..."
    pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "✅ PyTorch: $TORCH_VERSION"

# 安装其他依赖
echo "📦 安装其他依赖..."
pip install -q -r requirements.txt

echo ""
echo "========================================================"
echo "依赖安装完成！"
echo "========================================================"
echo ""

# 选择模式
echo "选择训练模式:"
echo "  1) 合成数据测试 (快速验证，~5 分钟)"
echo "  2) 真实 CSL 数据训练 (需要已下载数据集)"
echo "  3) 仅测试数据加载器"
echo ""
read -p "请输入选项 (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 使用合成数据训练..."
        echo "========================================================"
        python3 train.py \
            --use-synthetic \
            --epochs 10 \
            --batch-size 4 \
            --num-frames 16 \
            --frame-size 112 \
            --num-classes 100 \
            --num-workers 0
        ;;
    2)
        DATA_DIR="${DATA_DIR:-./data/CSL}"
        echo ""
        echo "🚀 使用真实数据训练..."
        echo "数据集目录：$DATA_DIR"
        echo "========================================================"
        
        if [ ! -d "$DATA_DIR" ]; then
            echo "❌ 错误：数据集目录不存在：$DATA_DIR"
            echo ""
            echo "请先下载 CSL 数据集:"
            echo "  1. 访问 https://github.com/ustc-slr/SLRDataset"
            echo "  2. 按指引申请下载"
            echo "  3. 解压到 $DATA_DIR"
            exit 1
        fi
        
        python3 train.py \
            --data-dir "$DATA_DIR" \
            --epochs 50 \
            --batch-size 4 \
            --num-workers 2
        ;;
    3)
        echo ""
        echo "🧪 测试数据加载器..."
        echo "========================================================"
        python3 dataset_loader.py
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "========================================================"
echo "✅ 训练完成！"
echo "========================================================"
echo ""
echo "检查点保存在：$SCRIPT_DIR/checkpoints/"
echo "日志保存在：$SCRIPT_DIR/logs/"
echo ""
echo "查看 TensorBoard:"
echo "  tensorboard --logdir ./logs --host 0.0.0.0 --port 6006"
echo ""
