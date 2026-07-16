# Memento-x-tool — GPU Worker 视频处理管线
# 9 节点视频主体替换
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

LABEL maintainer="Memento-X Team"
LABEL description="Memento GPU Worker — 9-node video subject replacement pipeline"

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git wget ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python3 \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# pip 镜像
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --upgrade pip

# ComfyUI
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /root/data/ComfyUI
WORKDIR /root/data/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt

# SAM2.1（Apache 2.0，完全开放）
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /root/data/sam2 \
    && pip install --no-cache-dir -e /root/data/sam2

# Memento 工具链
WORKDIR /root/data/memento-tool
COPY . .

# 模型下载
RUN bash download_models.sh

# 启动
CMD ["python3", "memento_pipeline/worker.py"]
