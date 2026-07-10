# ============================================================
# Memento-x-tool Docker 镜像
# 基础: nvidia/cuda:12.1-runtime-ubuntu22.04
# 包含: ComfyUI headless + 9 个 custom_nodes + 所有模型权重
# 用户 pull 后可直接使用，运行时无任何下载操作
# ============================================================

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

LABEL org.memento.name="Memento-x-tool"
LABEL org.memento.version="v1.0.0"
LABEL org.memento.description="ComfyUI headless + 9-node VFX pipeline (LTX-Video 2.3 + IC-LoRA)"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE_DIR=/workspace
ENV COMFYUI_DIR=/ComfyUI
ENV MODELS_DIR=/models
ENV CUSTOM_NODES_DIR=/ComfyUI/custom_nodes

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 3.10 设为默认 ──
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

# ── 工作目录 ──
WORKDIR /workspace

# ── 克隆 ComfyUI ──
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"

# ── 安装 ComfyUI 依赖 ──
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r "$COMFYUI_DIR/requirements.txt" \
    && pip install mediapipe huggingface_hub opencv-python-headless numpy Pillow

# ── 安装 SAM3（pip 安装） ──
# 注意：SAM3 模型权重需要提前下载到 /models/sam3/sam3.safetensors
# 用户在 HuggingFace facebook/sam3 申请访问权限后通过 download_models.sh 下载
RUN pip install sam3 --no-build-isolation

# ── 复制模型权重（预下载好的） ──
COPY ./models/ "$MODELS_DIR/"

# ── 模型软链接到 ComfyUI 预期路径 ──
RUN mkdir -p "$COMFYUI_DIR/models/checkpoints" \
    && mkdir -p "$COMFYUI_DIR/models/sams" \
    && mkdir -p "$COMFYUI_DIR/models/controlnet" \
    && mkdir -p "$COMFYUI_DIR/models/vae" \
    && mkdir -p "$COMFYUI_DIR/models/clip" \
    && mkdir -p "$COMFYUI_DIR/models/upscale_models" \
    && mkdir -p "$COMFYUI_DIR/models/diffusers"

# ── 复制 custom_nodes ──
COPY ./custom_nodes/ "$CUSTOM_NODES_DIR/"

# ── 安装 custom_nodes 依赖（各节点自己的 requirements） ──
RUN for node_dir in "$CUSTOM_NODES_DIR"/*/; do \
        if [ -f "$node_dir/requirements.txt" ]; then \
            pip install -r "$node_dir/requirements.txt"; \
        fi; \
    done

# ── 暴露端口 ──
EXPOSE 8188

# ── 启动命令 ──
CMD ["python3", "/ComfyUI/main.py", "--listen", "0.0.0.0", "--port", "8188"]