#!/bin/bash
# ============================================================
# Memento 模型下载脚本
# 下载所有 9 节点管线所需的模型权重到 ./models/
# 重试 3 次，失败则退出
# ============================================================
set -euo pipefail

MODEL_DIR="./models"
MAX_RETRIES=3
LOG_FILE="./download_models.log"

mkdir -p "$MODEL_DIR"/{sam3,mediapipe,motionbert,ltx,iclora,raft}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
retry() {
    local n=1
    local cmd="$*"
    until $cmd; do
        ((n++))
        if [ $n -gt $MAX_RETRIES ]; then
            log "ERROR: 重试 $MAX_RETRIES 次后仍然失败: $cmd"
            return 1
        fi
        log "RETRY ($n/$MAX_RETRIES): $cmd"
        sleep 5
    done
}

# ── 1. 安装 huggingface_hub ──
log "安装 huggingface_hub..."
pip install --quiet huggingface_hub

# ── 2. SAM3 (HuggingFace) ──
log "下载 SAM3 模型权重..."
# 注意：SAM3 需要先在 HuggingFace 上申请访问权限
# https://huggingface.co/facebook/sam3
retry python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'facebook/sam3',
    local_dir='$MODEL_DIR/sam3',
    local_dir_use_symlinks=False,
    allow_patterns=['sam3.safetensors', '*.json', '*.txt', '*.yaml']
)
" && log "SAM3 ✓" || { log "SAM3 下载失败"; exit 1; }

# ── 3. MediaPipe ──
log "安装 MediaPipe (无权重文件，仅 pip)..."
retry pip install --quiet mediapipe
log "MediaPipe ✓"

# ── 4. MotionBERT (GitHub Release) ──
log "下载 MotionBERT..."
MOTIONBERT_URL="https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/motionbert_ft_h36m.pth"
retry wget -q --show-progress -O "$MODEL_DIR/motionbert/motionbert_ft_h36m.pth" "$MOTIONBERT_URL" \
    && log "MotionBERT ✓" || { log "MotionBERT 下载失败"; exit 1; }

# ── 5. LTX-Video 2.3 主模型 (HuggingFace) ──
log "下载 LTX-Video 2.3 主模型 (约 10GB，FP8 量化)..."
# 主模型: ltx-2.3-22b-dev-fp8.safetensors ~10GB
retry python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Lightricks/LTX-2.3-fp8',
    filename='ltx-2.3-22b-dev-fp8.safetensors',
    local_dir='$MODEL_DIR/ltx',
    local_dir_use_symlinks=False,
)
" && log "LTX-Video 2.3 主模型 ✓" || { log "LTX-Video 2.3 下载失败"; exit 1; }

# ── 6. IC-LoRA Union Control 量化三合一 (HuggingFace) ──
log "下载 IC-LoRA Union Control (depth+canny+pose 三合一，约 654MB)..."
retry python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control',
    filename='ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
    local_dir='$MODEL_DIR/iclora',
    local_dir_use_symlinks=False,
)
" && log "IC-LoRA Union Control ✓" || { log "IC-LoRA Union Control 下载失败"; exit 1; }

# ── 7. IC-LoRA Ingredients 角色一致性 (HuggingFace) ──
log "下载 IC-LoRA Ingredients (角色一致性 Reference Sheet 约束)..."
retry python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients',
    filename='ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
    local_dir='$MODEL_DIR/iclora',
    local_dir_use_symlinks=False,
)
" && log "IC-LoRA Ingredients ✓" || { log "IC-LoRA Ingredients 下载失败"; exit 1; }

# ── 8. RAFT 光流模型 (torchvision 预训练权重) ──
log "预下载 RAFT 光流模型权重..."
# torchvision 的 raft_large C_T_V2 权重会自动下载到 torch hub 缓存
# 这里手动触发下载，确保构建镜像时已缓存
retry python3 -c "
import torch
import torchvision
print('预加载 RAFT Large 模型权重...')
model = torchvision.models.optical_flow.raft_large(
    weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2
)
print('RAFT 权重下载完成')
" && log "RAFT ✓" || { log "RAFT 下载失败"; exit 1; }

# ── 完成 ──
log "═══════════════════════════════════════"
log "模型下载完成！"
log "总大小: $(du -sh $MODEL_DIR | cut -f1)"
log "各模型:"
du -sh "$MODEL_DIR"/*/ | tee -a "$LOG_FILE"
log "═══════════════════════════════════════"