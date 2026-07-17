# ============================================================
# Memento ComfyUI GPU Worker — Docker 镜像
# 包含: ComfyUI + Python 依赖 + SAM2 + Memento 工具链
# 不含: 模型文件（通过 volume 挂载 /root/data/models）
#
# 英博云推送:
#   registry-cn-huabei1-internal.ebcloud.com/tenant-29013702/memento-comfyui
# ============================================================

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

LABEL maintainer="Memento-X Team"
LABEL description="Memento ComfyUI GPU Worker with 9-node pipeline"

# 国内镜像
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV HF_ENDPOINT=https://hf-mirror.com

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python3 \
    && ln -s /usr/bin/python3.10 /usr/bin/python

RUN pip install --upgrade pip

# ── ComfyUI ──
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/data/ComfyUI
WORKDIR /root/data/ComfyUI
# 降级 transformers（清华镜像可能未同步最新版）
RUN sed -i 's/transformers>=4.50.3/transformers>=4.49.0/' requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

# ── SAM2 ──
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /root/data/sam2 \
    && pip install --no-cache-dir -e /root/data/sam2

# ── Memento 工具链 + 自定义节点 ──
COPY . /root/data/Memento-x-tool
WORKDIR /root/data/Memento-x-tool

# 安装启动器依赖 + 自定义节点依赖
RUN pip install --no-cache-dir -r launcher/requirements.txt \
    && pip install --no-cache-dir mediapipe opencv-python-headless numpy

# 自定义节点软链接到 ComfyUI
RUN mkdir -p /root/data/ComfyUI/custom_nodes \
    && for d in /root/data/Memento-x-tool/custom_nodes/*/; do \
         name=$(basename "$d"); \
         ln -sf "$d" "/root/data/ComfyUI/custom_nodes/$name"; \
       done

# 模型目录（运行时挂载 host 的 /root/data/models）
RUN mkdir -p /root/data/models

EXPOSE 8188
WORKDIR /root/data/ComfyUI
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
