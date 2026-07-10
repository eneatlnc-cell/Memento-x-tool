"""Memento 启动器 v2.0 — 三方联通版主入口

双击运行后：
1. 读取/创建 config.json
2. 启动本地 API 服务（127.0.0.1:8189）
3. 检测 Docker / GPU / 镜像状态
4. 向云端中枢注册并维持心跳
5. 系统托盘常驻
"""
import logging
import os
import sys
import threading
import time
import signal
from pathlib import Path

# ── 确保 launcher 包可导入 ──
_launcher_dir = Path(__file__).parent
if str(_launcher_dir) not in sys.path:
    sys.path.insert(0, str(_launcher_dir.parent))

from launcher.config import (
    LauncherConfig, load_config, save_config, init_dirs, CONFIG_PATH, LOG_DIR
)
from launcher.docker_manager import DockerManager
from launcher.cloud_client import CloudClient
from launcher.local_server import app, setup_state, LogHandler
from launcher.tray import SystemTray

# ── 日志配置 ──
init_dirs()

log_handler = LogHandler()
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))

file_handler = logging.FileHandler(
    LOG_DIR / "launcher.log", encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[log_handler, file_handler],
)

logger = logging.getLogger("memento")


class LauncherApp:
    """启动器主应用"""

    def __init__(self):
        self.cfg = load_config()
        self.docker = DockerManager(self.cfg)
        self.cloud = CloudClient(self.cfg)
        self.tray: SystemTray = None
        self._server_thread: threading.Thread = None
        self._running = False

    def setup(self) -> bool:
        """环境检查"""
        logger.info("=" * 50)
        logger.info(f"Memento 启动器 v{self.cfg.version}")
        logger.info(f"配置文件: {CONFIG_PATH}")
        logger.info("=" * 50)

        # 1. 检查 Docker
        if not self.docker.check_docker_running():
            logger.error("Docker 未运行，请先启动 Docker Desktop")
            return False

        logger.info("Docker: ✓")

        # 2. 检查 GPU
        gpu = self.docker.get_gpu_info()
        if not gpu.get("available"):
            logger.error("GPU 不可用，请确认 NVIDIA 驱动已安装")
            return False

        self.cfg.gpu_info = gpu
        if gpu.get("vram_gb", 0) < 8:
            logger.error(f"显存不足: {gpu.get('vram_gb')} GB < 8 GB")
            return False

        logger.info(f"GPU: {gpu['model']} ({gpu['vram_gb']} GB) ✓")

        # 3. 检查镜像
        if self.docker.check_image_exists():
            logger.info(f"镜像: {self.cfg.docker_image} ✓")
        else:
            logger.info(f"镜像: {self.cfg.docker_image} (未拉取)")

        # 4. 检查容器
        container = self.docker.get_container()
        if container and container.status == "running":
            self.cfg.status = "running"
            self.cfg.container_id = container.short_id
            logger.info(f"容器: 运行中 ({container.short_id})")
        else:
            self.cfg.status = "idle"
            logger.info("容器: 未运行")

        return True

    def start_server(self):
        """启动本地 API 服务"""
        import uvicorn

        setup_state(self.cfg, self.docker, self.cloud)

        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.cfg.local_port,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(config)

        def run():
            logger.info(f"本地 API 服务已启动: http://127.0.0.1:{self.cfg.local_port}")
            server.run()

        self._server_thread = threading.Thread(target=run, daemon=True, name="api-server")
        self._server_thread.start()

        # 等待服务就绪
        time.sleep(2)

    def register(self):
        """向云端注册并启动心跳"""
        if not self.cfg.user_token and not self.cfg.user_id:
            logger.warning("未配置 user_token 或 user_id，跳过注册")
            logger.warning("请通过 Web 端或 POST /config 设置 Token")
            return

        if self.cloud.register():
            self.cloud.start_heartbeat()
            logger.info("云端注册成功，心跳已启动")
        else:
            logger.warning("云端注册失败，将在心跳时重试")
            self.cloud.start_heartbeat()

    def run(self):
        """主入口"""
        self._running = True

        if not self.setup():
            logger.error("环境检查未通过，启动器将以受限模式运行")
            self.cfg.status = "error"

        self.start_server()
        self.register()

        # 启动系统托盘
        def on_quit():
            self.shutdown()

        self.tray = SystemTray(self.cfg, self.docker, self.cloud, on_quit=on_quit)

        # 信号处理
        def signal_handler(sig, frame):
            logger.info(f"收到信号 {sig}，退出...")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info(f"启动器就绪 (状态: {self.cfg.status})")
        logger.info(f"Web 端可通过 http://127.0.0.1:{self.cfg.local_port} 访问")

        # 阻塞在托盘
        try:
            self.tray.start()
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """优雅关闭"""
        if not self._running:
            return
        self._running = False

        logger.info("正在关闭启动器...")

        self.cloud.stop_heartbeat()
        self.cloud.unregister()

        # 不停止容器，让用户手动决定
        # self.docker.stop_container()

        self.tray.stop()
        logger.info("启动器已关闭")
        sys.exit(0)


def main():
    app = LauncherApp()
    app.run()


if __name__ == "__main__":
    main()