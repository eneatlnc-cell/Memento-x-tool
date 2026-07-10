# Memento-x-tool

Memento 启动器 — ComfyUI headless + 9 节点 VFX 管线 Docker 镜像。

用户在自己 GPU 机器上一键启动，自动注册到 Memento 云端中枢，接收任务，渲染成片。

## 快速开始

```bash
# 1. 拉取镜像
docker pull mementoweb/memento-tool:v1.0.0

# 2. 设置 Token
export MEMENTO_USER_TOKEN="你的 JWT Token"

# 3. 启动
docker run --gpus all -d \
  --name memento-tool \
  -p 8188:8188 \
  -v ~/.memento:/workspace \
  -e MEMENTO_USER_TOKEN="$MEMENTO_USER_TOKEN" \
  mementoweb/memento-tool:v1.0.0
```

或使用启动器脚本：

```bash
python3 launcher.py --token "你的 JWT Token"
```

## 硬件要求

- GPU 显存 ≥ 8GB（推荐 RTX 4090 / A40，12GB+）
- CUDA 12.1+
- 系统内存 ≥ 16GB
- 磁盘 ≥ 50GB（模型 + 缓存）

## 构建镜像

```bash
# 1. 下载模型
bash download_models.sh

# 2. 构建
docker build -t mementoweb/memento-tool:v1.0.0 .
docker build -t mementoweb/memento-tool:latest .

# 3. 推送
docker push mementoweb/memento-tool:v1.0.0
docker push mementoweb/memento-tool:latest
```

## 镜像内容

- ComfyUI headless（官方最新稳定版）
- 9 个 Memento custom_nodes（管线骨架）
- 所有模型权重预置（SAM3 / MotionBERT / Wan3-DiT / RAFT）
- 无运行时下载操作

## 目录结构

```
Memento-x-tool/
├── Dockerfile
├── download_models.sh
├── launcher.py
├── custom_nodes/          # 9 个管线节点
│   ├── memento_01_preprocess/
│   ├── memento_02_segment/
│   ├── memento_03_pose2d/
│   ├── memento_04_pose3d/
│   ├── memento_05_quadmask/
│   ├── memento_06_wan3/
│   ├── memento_07_raft/
│   ├── memento_08_grading/
│   └── memento_09_export/
└── CONTEXT.md
```