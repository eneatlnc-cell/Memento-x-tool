#!/bin/bash
# ============================================================
# Memento 模型下载脚本（GPU 服务器端）
# 自动下载（国内高速镜像）+ 手动下载指引
# 单个失败不中断，继续下载其余
#
# 用法:
#   bash download_models.sh
#   MODEL_DIR=/data/models bash download_models.sh
#
# 详细说明: 见 MODEL_GUIDE.md
# ============================================================
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-/root/data/models}"
MAX_RETRIES=3
LOG_FILE="$MODEL_DIR/download.log"

# 国内镜像源
export HF_ENDPOINT="https://hf-mirror.com"

# pip 国内镜像（防止 pip install 走 pypi.org 海外，opencv-python/mediapipe 等包默认连接海外）
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"

mkdir -p "$MODEL_DIR"/{sam2,mediapipe,pose,ltx,iclora,raft}
mkdir -p "$MODEL_DIR/manual"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

retry() {
    local desc="$1"; shift; local n=1
    until "$@"; do
        ((n++))
        [ $n -gt $MAX_RETRIES ] && { log "⚠ $desc 失败，跳过（可手动下载）"; return 1; }
        log "  RETRY ($n/$MAX_RETRIES): $desc"; sleep 5
    done
    log "  ✓ $desc"
}

# ═══════════════════════════════════════════════════════════
# 自动下载（国内高速镜像）
# ═══════════════════════════════════════════════════════════
log "══════════════════════════════════════════"
log "  Memento 模型自动下载"
log "  镜像源: ${HF_ENDPOINT}"
log "  目标目录: ${MODEL_DIR}"
log "══════════════════════════════════════════"

# 1. MotionBERT — 162MB，hf-mirror
log "━━━ 1/6 MotionBERT 姿态估计 (162 MB) ━━━"
retry "MotionBERT" python3 -c "
import os; os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT}'
from huggingface_hub import snapshot_download
snapshot_download('walterzhu/MotionBERT', local_dir='${MODEL_DIR}/pose',
    local_dir_use_symlinks=False, resume_download=True)
" 2>/dev/null || \
retry "MotionBERT-wget" wget -q --show-progress --continue \
    -O "$MODEL_DIR/pose/motionbert_ft_h36m.pth" \
    "${HF_ENDPOINT}/walterzhu/MotionBERT/resolve/main/motionbert_ft_h36m.pth" || true

# 2. IC-LoRA Union Control — 654MB，开放下载
log "━━━ 2/6 IC-LoRA Union Control (654 MB) ━━━"
retry "UnionControl" python3 -c "
import os; os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT}'
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control',
    'ltx-2.3-22b-ic-lora-union-control-0.9.safetensors',
    local_dir='${MODEL_DIR}/iclora', local_dir_use_symlinks=False, resume_download=True)
" 2>/dev/null || retry "UnionControl-wget" wget -q --show-progress --continue     -O "${MODEL_DIR}/iclora/ltx-2.3-22b-ic-lora-union-control-0.9.safetensors"     "${HF_ENDPOINT}/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-0.9.safetensors" || true

# 3. LTX-2.3 FP8 — 29GB，优先 ModelScope 国内 CDN
log "━━━ 3/6 LTX-2.3 FP8 主模型 (29 GB) ━━━"
log "  尝试 ModelScope 国内 CDN..."
if command -v modelscope &>/dev/null || pip install --quiet --no-deps modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null; then
    retry "LTX-ModelScope" python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('Lightricks/LTX-2.3-fp8', local_dir='${MODEL_DIR}/ltx',
    allow_patterns=['*dev-fp8*'], resume_download=True)
" 2>/dev/null && log "  ✓ LTX 从 ModelScope 下载完成" || true
fi

# ModelScope 失败则回退 hf-mirror
if [ ! -f "$MODEL_DIR/ltx/ltx-2.3-22b-dev-fp8.safetensors" ]; then
    log "  ModelScope 不可用，改用 hf-mirror..."
    retry "LTX-HF" python3 -c "
import os; os.environ['HF_ENDPOINT'] = '${HF_ENDPOINT}'
from huggingface_hub import hf_hub_download
hf_hub_download('Lightricks/LTX-2.3-fp8', 'ltx-2.3-22b-dev-fp8.safetensors',
    local_dir='${MODEL_DIR}/ltx', local_dir_use_symlinks=False, resume_download=True)
" 2>/dev/null ||     retry "LTX-HF-wget" wget -q --show-progress --continue         -O "${MODEL_DIR}/ltx/ltx-2.3-22b-dev-fp8.safetensors"         "${HF_ENDPOINT}/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors" || true
fi

# 4. MediaPipe
log "━━━ 4/6 MediaPipe ━━━"
retry "MediaPipe" pip install --no-deps --quiet mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install --quiet mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple || true

# 5. RAFT 光流
log "━━━ 5/6 RAFT 光流模型 (~50 MB) ━━━"
retry "RAFT" python3 -c "
import torch, torchvision
torchvision.models.optical_flow.raft_large(weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2)
print('RAFT OK')
" || true

# 6. SAM2 源码
log "━━━ 6/6 SAM2 源码 ━━━"
if [ ! -d "/root/data/sam2" ]; then
    retry "SAM2源码" git clone --depth 1 https://github.com/facebookresearch/sam2.git /root/data/sam2 2>/dev/null || true
    pip install -e /root/data/sam2 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || pip install --quiet -e /root/data/sam2 -i https://pypi.tuna.tsinghua.edu.cn/simple || true
fi

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
echo ""
log "══════════════════════════════════════════"
log "  自动下载完成"
log "══════════════════════════════════════════"

echo ""
echo "  自动下载状态:"
for m in pose ltx iclora raft; do
    size=$(du -sh "$MODEL_DIR/$m" 2>/dev/null | cut -f1)
    case "$size" in
        ""|"0"|"4.0K") echo "  ✗ $m — 未就绪" ;;
        *) echo "  ✓ $m — $size" ;;
    esac
done
echo ""

# ═══════════════════════════════════════════════════════════
# 手动下载指引
# ═══════════════════════════════════════════════════════════
cat << 'MANUAL'

  ╔══════════════════════════════════════════════════════════╗
  ║              ⚠ 需要手动下载的模型                        ║
  ║         详细说明请参阅 MODEL_GUIDE.md                    ║
  ╚══════════════════════════════════════════════════════════╝

  ┌─ 1. SAM2.1 权重 (898 MB) ─────────────────────────────┐
  │ 下载: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  │ 放置: ${MODEL_DIR}/sam2/sam2.1_hiera_large.pt
  │ 原因: Meta 美国 CDN，国内直连较慢
  └────────────────────────────────────────────────────────┘

  ┌─ 2. IC-LoRA Ingredients (1.31 GB) — Gated ────────────┐
  │ ① 注册: https://huggingface.co/join
  │ ② 授权: https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients
  │    点击 "Agree and access repository"
  │ ③ 创建 Token: https://huggingface.co/settings/tokens
  │ ④ 设置环境变量后下载:
  │    export HF_TOKEN="hf_你的token"
  │    export HF_ENDPOINT=https://hf-mirror.com
  │    python3 -c "
  │    from huggingface_hub import hf_hub_download
  │    hf_hub_download('Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients',
  │      'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors',
  │      local_dir='${MODEL_DIR}/iclora', local_dir_use_symlinks=False)
  │    "
  └────────────────────────────────────────────────────────┘

  ┌─ 3. MediaPipe 模型文件 (~15 MB) ──────────────────────┐
  │ 原因: 首次运行从 Google 下载，国内可能被墙
  │ 解决: 设置代理或使用 VPN 后运行一次
  └────────────────────────────────────────────────────────┘

  完整模型目录结构:
  ${MODEL_DIR}/
  ├── sam2/sam2.1_hiera_large.pt         ← 手动
  ├── iclora/
  │   ├── union-control-*.safetensors    ← 自动
  │   └── ingredients-*.safetensors      ← 手动
  ├── ltx/ltx-2.3-22b-dev-fp8.safetensors ← 自动
  ├── pose/motionbert_ft_h36m.pth        ← 自动
  └── raft/                              ← 自动

MANUAL
