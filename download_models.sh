#!/bin/bash
# ============================================================
# Memento 模型下载脚本
# 下载所有 9 节点管线所需的模型权重到 ./models/
# 使用 HuggingFace 镜像 hf-mirror.com 加速
# 重试 3 次，失败则退出
# ============================================================
set -euo pipefail

MODEL_DIR="./models"
MAX_RETRIES=3
LOG_FILE="./download_models.log"

mkdir -p "$MODEL_DIR"/{sam3,mediapipe,motionbert,ltx,iclora,raft}

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

# ── 1. 安装 huggingface_hub ──
log "安装 huggingface_hub..."
pip install --quiet huggingface_hub

# ── 2. SAM3 ──
log "下载 SAM3 模型权重..."
retry "SAM3" python3 -c "$(cat << 'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    "facebook/sam3",
    local_dir="./models/sam3",
    local_dir_use_symlinks=False,
    allow_patterns=["sam3.safetensors", "*.json", "*.txt", "*.yaml"],
)
PYEOF
)"

# ── 3. MediaPipe ──
log "安装 MediaPipe..."
retry "MediaPipe" pip install --quiet mediapipe

# ── 4. MotionBERT ──
log "下载 MotionBERT..."
MOTIONBERT_URL="https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/motionbert_ft_h36m.pth"
retry "MotionBERT" wget -q --show-progress -O "$MODEL_DIR/motionbert/motionbert_ft_h36m.pth" "$MOTIONBERT_URL"

# ── 5. LTX-Video 2.3 主模型 ──
log "下载 LTX-Video 2.3 主模型 (约 10GB，FP8 量化)..."
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

# ── 6. IC-LoRA Union Control ──
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

# ── 7. IC-LoRA Ingredients ──
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

# ── 8. RAFT 光流模型 ──
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
