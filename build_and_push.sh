#!/bin/bash
# ============================================================
# Memento Docker 镜像构建 + 推送到英博云
# 在 GPU 云实例上执行一次，后续客户直接拉取使用
# ============================================================
set -euo pipefail

# ── 英博云镜像仓库配置 ──
REGISTRY="registry-cn-huabei1-internal.ebcloud.com/tenant-29013702"
IMAGE="memento-comfyui"
TAG="${TAG:-latest}"

# ── 登录（首次需要） ──
echo ">>> 登录英博云镜像仓库..."
docker login "${REGISTRY%%/*}" -u u1014328-cn-huabei1 --password-stdin <<< "你的密码"

# ── 构建 ──
echo ">>> 构建 Docker 镜像..."
docker build -t "${REGISTRY}/${IMAGE}:${TAG}" .

# ── 推送 ──
echo ">>> 推送到英博云..."
docker push "${REGISTRY}/${IMAGE}:${TAG}"

echo ">>> 完成！镜像: ${REGISTRY}/${IMAGE}:${TAG}"
echo ">>> 后续在任意实例拉取: docker pull ${REGISTRY}/${IMAGE}:${TAG}"
