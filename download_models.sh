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

mkdir -p "$MODEL_DIR"/{sam3,mediapipe,motionbert,wan3,vace3,raft}

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

# ── 2. SAM3-Large (HuggingFace) ──
log "下载 SAM3-Large..."
retry python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'facebook/sam2-hiera-large',
    local_dir='$MODEL_DIR/sam3',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.msgpack']
)
" && log "SAM3-Large ✓" || { log "SAM3 下载失败"; exit 1; }

# ── 3. MediaPipe ──
log "安装 MediaPipe (无权重文件，仅 pip)..."
retry pip install --quiet mediapipe
log "MediaPipe ✓"

# ── 4. MotionBERT (GitHub Release) ──
log "下载 MotionBERT..."
MOTIONBERT_URL="https://github.com/Walter0807/MotionBERT/releases/download/v1.0.0/motionbert_ft_h36m.pth"
retry wget -q --show-progress -O "$MODEL_DIR/motionbert/motionbert_ft_h36m.pth" "$MOTIONBERT_URL" \
    && log "MotionBERT ✓" || { log "MotionBERT 下载失败"; exit 1; }

# ── 5. Wan3-DiT (HuggingFace) ──
log "下载 Wan3-DiT (约 14GB，耗时较长)..."
retry python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.1-T2V-14B',
    local_dir='$MODEL_DIR/wan3',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.msgpack', '*.safetensors.index.json']
)
" && log "Wan3-DiT ✓" || { log "Wan3 下载失败"; exit 1; }

# ── 6. VACE3 ──
log "VACE3: 检查是否开源..."
VACE3_STATUS="NOT_OPEN_SOURCE"
log "VACE3 当前非开源模型，留空目录 $MODEL_DIR/vace3/"
echo "$VACE3_STATUS" > "$MODEL_DIR/vace3/STATUS.txt"
log "VACE3 ⚠ (留空，待后续确认)"

# ── 7. RAFT (HuggingFace) ──
log "下载 RAFT..."
retry python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'pytorch/raft_large',
    local_dir='$MODEL_DIR/raft',
    local_dir_use_symlinks=False
)
" && log "RAFT ✓" || { log "RAFT 下载失败"; exit 1; }

# ── 完成 ──
log "═══════════════════════════════════════"
log "模型下载完成！"
log "总大小: $(du -sh $MODEL_DIR | cut -f1)"
log "各模型:"
du -sh "$MODEL_DIR"/*/ | tee -a "$LOG_FILE"
log "═══════════════════════════════════════"