#!/bin/bash
# ============================================================
# Memento 模型下载脚本
# 全部走国内镜像，兼容 GitHub 被墙环境
# 单个模型失败不中断，继续下载其余模型
# ============================================================
set -uo pipefail

MODEL_DIR="./models"
MAX_RETRIES=3
LOG_FILE="./download_models.log"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
GH_MIRROR="${GH_MIRROR:-https://gitclone.com/github.com}"

mkdir -p "$MODEL_DIR"/{sam2,mediapipe,motionbert,ltx,iclora,raft}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

retry() {
    local desc="$1"; shift
    local n=1
    until "$@"; do
        ((n++))
        if [ $n -gt $MAX_RETRIES ]; then
            log "WARNING: $desc 失败，跳过（不影响整体）"
            return 1
        fi
        log "RETRY ($n/$MAX_RETRIES): $desc"
        sleep 5
    done
    log "$desc ✓"
    return 0
}

ensure_hf_hub() {
    python3 -c "import huggingface_hub" 2>/dev/null && return 0
    log "安装 huggingface_hub（--no-deps 避免重装 PyTorch）..."
    pip install --no-deps --quiet huggingface_hub filelock fsspec pyyaml requests tqdm typing-extensions packaging
}

# ── 1. SAM2.1 源码 + 权重 ──
log "━━━ 1/7 SAM2.1 视频分割 ━━━"
if [ ! -d "/opt/sam2" ]; then
    log "克隆 SAM2 源码..."
    retry "SAM2源码" git clone --depth 1 "$GH_MIRROR/facebookresearch/sam2.git" /opt/sam2 || true
    pip install --no-deps --quiet -e /opt/sam2 2>/dev/null || pip install --quiet -e /opt/sam2 || true
fi
retry "SAM2权重" wget -q --show-progress -O "$MODEL_DIR/sam2/sam2.1_hiera_large.pt" \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" || true

# ── 2. MediaPipe ──
log "━━━ 2/7 MediaPipe ━━━"
retry "MediaPipe" pip install --no-deps --quiet mediapipe 2>/dev/null || pip install --quiet mediapipe || true

# ── 3. MotionBERT（GitHub 镜像）──
log "━━━ 3/7 MotionBERT ━━━"
retry "MotionBERT" wget -q --show-progress -O "$MODEL_DIR/motionbert/motionbert_ft_h36m.pth" \
    "${GH_MIRROR}/Walter0807/MotionBERT/releases/download/v1.0.0/motionbert_ft_h36m.pth" || true

# ── 4. LTX-Video 2.3 主模型 ──
log "━━━ 4/7 LTX-Video 2.3 (约 10GB) ━━━"
ensure_hf_hub
retry "LTX" python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3-fp8','ltx-2.3-22b-dev-fp8.safetensors',
    local_dir='./models/ltx',local_dir_use_symlinks=False,resume_download=True)
" || true

# ── 5. IC-LoRA Union Control ──
log "━━━ 5/7 IC-LoRA Union Control (654MB) ━━━"
retry "Union Control" python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control','ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors',
    local_dir='./models/iclora',local_dir_use_symlinks=False,resume_download=True)
" || true

# ── 6. IC-LoRA Ingredients ──
log "━━━ 6/7 IC-LoRA Ingredients (1.5GB) ━━━"
retry "Ingredients" python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients','ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
    local_dir='./models/iclora',local_dir_use_symlinks=False,resume_download=True)
" || true

# ── 7. RAFT ──
log "━━━ 7/7 RAFT 光流 ━━━"
retry "RAFT" python3 -c "
import torch, torchvision
torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2)
print('RAFT OK')
" || true

# ── 完成 ──
echo ""
log "════════════════════════════════"
log " 模型下载完成"
log " 总大小: $(du -sh "$MODEL_DIR" 2>/dev/null | cut -f1)"
echo ""
for m in sam2 mediapipe motionbert ltx iclora raft; do
    size=$(du -sh "$MODEL_DIR/$m" 2>/dev/null | cut -f1)
    [ "$size" = "0" ] || [ -z "$size" ] && echo "  ✗ $m — 未就绪" || echo "  ✓ $m — $size"
done
log "════════════════════════════════"
