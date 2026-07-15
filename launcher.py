#!/usr/bin/env python3
"""Memento-x-tool 启动器

用户在自己 GPU 机器上运行此脚本，自动完成：
1. 拉取最新 Docker 镜像
2. 启动 ComfyUI headless 容器
3. 注册到云端中枢
4. 心跳保持

用法:
  python launcher.py --token YOUR_TOKEN
  python launcher.py --token YOUR_TOKEN --api-url http://memento.asia/api/v1
"""
import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memento-launcher")

# ── 默认值 ──
DEFAULT_API_URL = "http://memento.asia/api/v1"
DEFAULT_IMAGE = "mementoweb/memento-tool:v1.0.0"


def get_args():
    parser = argparse.ArgumentParser(description="Memento-x-tool 启动器")
    parser.add_argument("--api-url", default=os.getenv("MEMENTO_API_URL", DEFAULT_API_URL))
    parser.add_argument("--token", default=os.getenv("MEMENTO_USER_TOKEN", ""))
    parser.add_argument("--image", default=os.getenv("MEMENTO_IMAGE", DEFAULT_IMAGE))
    parser.add_argument("--workspace", default=os.getenv("MEMENTO_WORKSPACE", os.path.expanduser("~/.memento")))
    parser.add_argument("--port", default=os.getenv("MEMENTO_PORT", "8189"))
    parser.add_argument("--gpu", default=os.getenv("CUDA_VISIBLE_DEVICES", "0"))
    return parser.parse_args()


def main():
    args = get_args()
    if not args.token:
        logger.error("请设置 MEMENTO_USER_TOKEN 或使用 --token 参数")
        sys.exit(1)

    # 确保 launcher 包可导入
    launcher_dir = os.path.dirname(os.path.abspath(__file__))
    if launcher_dir not in sys.path:
        sys.path.insert(0, launcher_dir)

    from launcher.config import LauncherConfig, save_config

    # 写入配置
    cfg = LauncherConfig(
        api_url=args.api_url,
        user_token=args.token,
        docker_image=args.image,
        local_port=int(args.port),
        gpu_device=args.gpu,
        workspace=args.workspace,
    )
    save_config(cfg)

    # 启动 GUI 版
    from launcher.launcher_gui import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
