#!/usr/bin/env python3
"""
Memento-x-tool 启动器
=====================
用户在自己 GPU 机器上运行此脚本，自动完成：
1. 拉取最新 Docker 镜像（版本锁定）
2. 启动 ComfyUI headless 容器
3. 注册到 Memento 云端中枢
4. 心跳保持 + 状态回传
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memento-launcher")

# ── 配置 ──
DEFAULT_API_URL = "http://118.31.189.101:8000/api/v1"
DEFAULT_IMAGE = "mementoweb/memento-tool:v1.0.0"
DEFAULT_WORKSPACE = os.path.expanduser("~/.memento")
HEARTBEAT_INTERVAL = 30  # 秒


def get_config():
    """从环境变量或命令行获取配置"""
    parser = argparse.ArgumentParser(description="Memento-x-tool 启动器")
    parser.add_argument("--api-url", default=os.getenv("MEMENTO_API_URL", DEFAULT_API_URL))
    parser.add_argument("--token", default=os.getenv("MEMENTO_USER_TOKEN", ""))
    parser.add_argument("--image", default=os.getenv("MEMENTO_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--workspace", default=os.getenv("MEMENTO_WORKSPACE", DEFAULT_WORKSPACE))
    parser.add_argument("--port", default=os.getenv("MEMENTO_PORT", "8188"))
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    return parser.parse_args()


def check_docker():
    """检查 Docker 是否可用"""
    try:
        subprocess.run(["docker", "info"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker 未安装或不可用，请先安装 Docker")
        return False


def check_gpu():
    """检查 GPU 是否可用"""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            logger.info("GPU 检测通过: nvidia-smi 可用")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    logger.error("GPU 不可用，请确认 NVIDIA 驱动已安装")
    return False


def pull_image(image: str) -> bool:
    """拉取 Docker 镜像"""
    logger.info(f"拉取镜像: {image}")
    try:
        subprocess.run(["docker", "pull", image], check=True)
        logger.info("镜像拉取完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"镜像拉取失败: {e}")
        return False


def start_container(image: str, workspace: str, port: str, gpu: str) -> str | None:
    """启动 ComfyUI headless 容器"""
    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    for sub in ["assets", "workspace", "context_buffer", "outputs", "logs"]:
        (workspace_path / sub).mkdir(exist_ok=True)

    logger.info(f"启动容器: {image}")
    cmd = [
        "docker", "run", "-d",
        "--name", "memento-tool",
        "--gpus", f'"device={gpu}"',
        "-p", f"{port}:8188",
        "-v", f"{workspace}:/workspace",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
        image,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        container_id = result.stdout.strip()
        logger.info(f"容器已启动: {container_id[:12]}")
        return container_id
    except subprocess.CalledProcessError as e:
        logger.error(f"容器启动失败: {e.stderr}")
        return None


def api_request(api_url: str, path: str, token: str, data: dict | None = None) -> dict | None:
    """发送 HTTP 请求到 Memento API"""
    url = f"{api_url}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"API 请求失败: {path} — {e}")
        return None


def register(api_url: str, token: str) -> bool:
    """注册启动器到云端中枢"""
    logger.info("注册到云端中枢...")
    result = api_request(api_url, "/workflow/launcher/register", token, {"type": "comfyui_headless"})
    if result:
        logger.info("注册成功")
        return True
    return False


def heartbeat(api_url: str, token: str) -> bool:
    """发送心跳"""
    return api_request(api_url, "/workflow/launcher/heartbeat", token) is not None


def cleanup(api_url: str, token: str):
    """清理：注销 + 停止容器"""
    logger.info("正在清理...")
    api_request(api_url, "/workflow/launcher/unregister", token)
    subprocess.run(["docker", "stop", "memento-tool"], capture_output=True)
    subprocess.run(["docker", "rm", "memento-tool"], capture_output=True)
    logger.info("清理完成")


def main():
    args = get_config()

    if not args.token:
        logger.error("请设置 MEMENTO_USER_TOKEN 环境变量或使用 --token 参数")
        sys.exit(1)

    # 1. 环境检查
    if not check_docker():
        sys.exit(1)
    if not check_gpu():
        sys.exit(1)

    # 2. 拉取镜像
    if not pull_image(args.image):
        sys.exit(1)

    # 3. 启动容器
    container_id = start_container(args.image, args.workspace, args.port, args.gpu)
    if not container_id:
        sys.exit(1)

    # 4. 注册到中枢
    time.sleep(5)  # 等待 ComfyUI 启动
    if not register(args.api_url, args.token):
        logger.error("注册失败，终止")
        cleanup(args.api_url, args.token)
        sys.exit(1)

    # 5. 信号处理
    def signal_handler(sig, frame):
        logger.info(f"收到信号 {sig}，退出...")
        cleanup(args.api_url, args.token)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 6. 心跳循环
    logger.info(f"启动器运行中 (心跳间隔: {HEARTBEAT_INTERVAL}s)")
    while True:
        time.sleep(HEARTBEAT_INTERVAL)
        if not heartbeat(args.api_url, args.token):
            logger.warning("心跳失败，将在下次重试")
        else:
            logger.debug("心跳 OK")


if __name__ == "__main__":
    main()