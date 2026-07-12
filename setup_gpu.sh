#!/bin/bash
# ============================================================
# Memento GPU Worker 一键部署脚本
# 在 GPU 云实例上执行，自动完成所有环境配置
# 使用国内镜像源，兼容云GPU预装环境（conda PyTorch）
# ============================================================
set -euo pipefail

PIP_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
PIP_TRUST="mirrors.aliyun.com"
HF_MIRROR="https://hf-mirror.com"
COMFYUI_DIR="/opt/ComfyUI"
TOOL_DIR="/opt/memento-tool"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. 配置 pip 国内镜像 ──
log "配置 pip 阿里云镜像..."
python3 -m pip config set global.index-url "$PIP_MIRROR" 2>/dev/null || true
python3 -m pip config set global.trusted-host "$PIP_TRUST" 2>/dev/null || true
pip install --upgrade pip

# ── 2. 克隆 ComfyUI ──
log "克隆 ComfyUI..."
if [ -d "$COMFYUI_DIR" ]; then
    log "ComfyUI 已存在，跳过"
else
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
fi

# ── 3. 安装 ComfyUI 依赖（--no-deps 避免重装 PyTorch）──
log "安装 ComfyUI 依赖（跳过已有 PyTorch）..."
cd "$COMFYUI_DIR"
pip install --no-deps -r requirements.txt 2>/dev/null || pip install -r requirements.txt
# 补装漏掉的传递依赖
python3 -c "import cv2" 2>/dev/null || pip install opencv-python
python3 -c "import PIL" 2>/dev/null || pip install Pillow

# ── 4. 克隆 Memento 工具链 ──
log "克隆 Memento 工具链..."
if [ -d "$TOOL_DIR" ]; then
    log "工具链已存在，git pull 更新..."
    cd "$TOOL_DIR" && git pull
else
    git clone --depth 1 https://github.com/eneatlnc-cell/Memento-x-tool.git "$TOOL_DIR"
fi

# ── 5. 下载模型 ──
log "下载模型权重（HF 镜像: $HF_MIRROR）..."
cd "$TOOL_DIR"
export HF_ENDPOINT="$HF_MIRROR"
bash download_models.sh

log "============================================"
log "GPU Worker 部署完成！"
log "ComfyUI: $COMFYUI_DIR"
log "Memento:  $TOOL_DIR"
log "============================================"
