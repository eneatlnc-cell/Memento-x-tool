#!/bin/bash
# ============================================================
# Memento 模型下载脚本
# 下载所有 9 节点管线所需的模型权重到 ./models/
# 使用国内镜像源加速（HuggingFace / Meta 官方直链）
# 重试 3 次，失败则退出
# ============================================================
set -euo pipefail

MODEL_DIR="./models"
MAX_RETRIES=3
LOG_FILE="./download_models.log"

mkdir -p "$MODEL_DIR"/{sam2,mediapipe,motionbert,ltx,iclora,raft}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

retry() {
    local desc="$1"; shift
    local n=1
    until "$@"; do
        ((n++))
        if [ $n -gt $MAX_RETRIES ]; then
            log "ERROR: $desc 重试 $MAX_RETRIES 次后仍然失败"
            return 1
        fi
        log "RETRY ($n/$MAX_RETRIES): $desc"
        sleep 5
    done
    log "$desc ✓"
}

# ── 1. SAM2（Meta 官方直链，无需认证，完全开放）──
log "下载 SAM2 模型权重 (sam2.1_hiera_large.pt, 约 900MB)..."
if [ ! -d "/opt/sam2" ]; then
    log "克隆 SAM2 源码..."
    retry "SAM2源码" git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2
    pip install --quiet -e /opt/sam2
fi
SAM2_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
retry "SAM2权重" wget -q --show-progress -O "$MODEL_DIR/sam2/sam2.1_hiera_large.pt" "$SAM2_URL"

# ── 2. MediaPipe ──
log "安装 MediaPipe..."
retry "MediaPipe" pip install --quiet mediapipe

# ── 3. MotionBERT ──
log "下载 MotionBERT..."
MOTIONBERT_URL="https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/motionbert_ft_h36m.pth"
retry "MotionBERT" wget -q --show-progress -O "$MODEL_DIR/motionbert/motionbert_ft_h36m.pth" "$MOTIONBERT_URL"

# ── 4. LTX-Video 2.3 主模型 (HuggingFace) ──
log "下载 LTX-Video 2.3 主模型 (约 10GB，FP8 量化)..."
pip install --quiet huggingface_hub
retry "LTX-Video" python3 -c "$(cat << 'PYEOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Lightricks/LTX-2.3-fp8",
    filename="ltx-2.3-22b-dev-fp8.safetensors",
    local_dir="./models/ltx",
    local_dir_use_symlinks=False,
)
PYEOF
)"

# ── 5. IC-LoRA Union Control ──
log "下载 IC-LoRA Union Control (三合一控制，约 654MB)..."
retry "Union Control" python3 -c "$(cat << 'PYEOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
    filename="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    local_dir="./models/iclora",
    local_dir_use_symlinks=False,
)
PYEOF
)"

# ── 6. IC-LoRA Ingredients ──
log "下载 IC-LoRA Ingredients (角色一致性约束)..."
retry "Ingredients" python3 -c "$(cat << 'PYEOF'
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
    filename="ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
    local_dir="./models/iclora",
    local_dir_use_symlinks=False,
)
PYEOF
)"

# ── 7. RAFT 光流模型 ──
log "预下载 RAFT 光流模型权重..."
retry "RAFT" python3 -c "$(cat << 'PYEOF'
import torch, torchvision
print("预加载 RAFT Large...")
torchvision.models.optical_flow.raft_large(
    weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2
)
print("RAFT 权重下载完成")
PYEOF
)"

# ── 完成 ──
log "═══════════════════════════════════════"
log "模型下载完成！"
log "总大小: $(du -sh "$MODEL_DIR" | cut -f1)"
du -sh "$MODEL_DIR"/*/ | tee -a "$LOG_FILE"
log "═══════════════════════════════════════"
