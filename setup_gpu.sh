#!/bin/bash
# ============================================================
# Memento GPU Worker 一键部署脚本
# 兼容国内 GPU 云（GitHub 被墙），全部走国内镜像
# 用法: 新建 GPU 容器后，直接复制粘贴执行
# ============================================================
set -uo pipefail

PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
HF_MIRROR="https://hf-mirror.com"
GH_MIRROR="https://gitclone.com/github.com"    # GitHub 镜像
COMFYUI_DIR="/root/data/ComfyUI"
TOOL_DIR="/root/data/memento-tool"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. 配置 pip 国内镜像 ──
log "1/6 配置 pip 阿里云镜像..."
pip config set global.index-url "$PIP_MIRROR" 2>/dev/null || true
pip config set global.trusted-host "mirrors.aliyun.com" 2>/dev/null || true
pip install --quiet --upgrade pip

# ── 2. 克隆 ComfyUI（走 gitclone 镜像）──
log "2/6 克隆 ComfyUI..."
if [ -d "$COMFYUI_DIR" ]; then
    log "ComfyUI 已存在，跳过"
else
    git clone --depth 1 "$GH_MIRROR/comfyanonymous/ComfyUI.git" "$COMFYUI_DIR" 2>&1 | tail -1
fi

# ── 3. 安装 ComfyUI 依赖（--no-deps 避免重装 PyTorch）──
log "3/6 安装 ComfyUI 依赖..."
cd "$COMFYUI_DIR"
pip install --no-deps -r requirements.txt 2>/dev/null || pip install -r requirements.txt
python3 -c "import cv2" 2>/dev/null || pip install opencv-python
python3 -c "import PIL" 2>/dev/null || pip install Pillow

# ── 4. 克隆 Memento 工具链（走 gitclone 镜像）──
log "4/6 克隆 Memento 工具链..."
rm -rf "$TOOL_DIR"
git clone --depth 1 "$GH_MIRROR/eneatlnc-cell/Memento-x-tool.git" "$TOOL_DIR" 2>&1 | tail -1

# ── 5. 下载模型 ──
log "5/6 下载模型权重..."
cd "$TOOL_DIR"
export HF_ENDPOINT="$HF_MIRROR"
bash download_models.sh

# ── 6. 完成 ──
log "6/6 部署完成！"
log "ComfyUI: $COMFYUI_DIR"
log "Memento:  $TOOL_DIR"
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
