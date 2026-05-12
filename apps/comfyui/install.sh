#!/bin/bash
# LLaDA2.0_Uni ComfyUI 节点快速安装脚本

set -e

echo "=========================================="
echo "LLaDA2.0_Uni ComfyUI 节点安装"
echo "=========================================="

# 检查 ComfyUI 路径
if [ -z "$1" ]; then
    echo "用法: $0 /path/to/ComfyUI"
    echo ""
    echo "示例:"
    echo "  $0 ~/ComfyUI"
    exit 1
fi

COMFYUI_PATH="$1"
CUSTOM_NODES_DIR="$COMFYUI_PATH/custom_nodes"

if [ ! -d "$COMFYUI_PATH" ]; then
    echo "❌ ComfyUI 路径不存在: $COMFYUI_PATH"
    exit 1
fi

if [ ! -d "$CUSTOM_NODES_DIR" ]; then
    echo "❌ custom_nodes 目录不存在: $CUSTOM_NODES_DIR"
    exit 1
fi

# 获取当前脚本所在目录（即 apps/comfyui）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 验证项目结构完整（需要上层的 encoder/ 和 decoder/）
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ ! -d "$PROJECT_ROOT/encoder" ] || [ ! -d "$PROJECT_ROOT/decoder" ]; then
    echo "❌ 项目结构不完整，缺少 encoder/ 或 decoder/ 目录"
    echo "   项目根目录: $PROJECT_ROOT"
    echo ""
    echo "请确保通过 git clone 获取了完整项目:"
    echo "   git clone https://github.com/inclusionAI/LLaDA2.0-Uni.git"
    exit 1
fi

echo "📂 ComfyUI 路径: $COMFYUI_PATH"
echo "📂 节点源路径: $SCRIPT_DIR"
echo "📂 项目根目录: $PROJECT_ROOT"
echo ""

# 创建软链接
TARGET_DIR="$CUSTOM_NODES_DIR/LLaDA2Uni"

if [ -e "$TARGET_DIR" ]; then
    echo "⚠️  目标已存在: $TARGET_DIR"
    read -p "是否删除并重新创建？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
    else
        echo "❌ 取消安装"
        exit 1
    fi
fi

ln -s "$SCRIPT_DIR" "$TARGET_DIR"
echo "✅ 创建软链接: $TARGET_DIR -> $SCRIPT_DIR"

# 安装依赖
echo ""
echo "📦 安装 Python 依赖..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"
echo "✅ 依赖安装完成"

# 可选：Flash Attention
echo ""
read -p "是否安装 Flash Attention 2？(推荐，但编译较慢) (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 安装 Flash Attention 2..."
    pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention 安装失败，可以继续使用 SDPA"
fi

echo ""
echo "=========================================="
echo "✅ 安装完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 启动 ComfyUI:"
echo "   cd $COMFYUI_PATH && python main.py"
echo ""
echo "2. 在浏览器打开: http://localhost:8188"
echo ""
echo "3. 右键 → Add Node → LLaDA2.0_Uni"
echo ""
echo "4. 在 Loader 节点中设置模型路径:"
echo "   inclusionAI/LLaDA2.0-Uni"
echo "   （首次使用会自动从 HuggingFace 下载，也可填写本地路径）"
echo ""
echo "详细文档: $SCRIPT_DIR/README.md"
