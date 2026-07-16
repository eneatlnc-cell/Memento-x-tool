"""Memento 启动器 — 系统托盘

使用 pystray 实现 Windows 系统托盘：
- 状态图标：绿（在线）/ 黄（任务中）/ 灰（离线）
- 菜单：查看日志、重启容器、退出
"""
import logging
import threading
import sys
from pathlib import Path

logger = logging.getLogger("memento.tray")

try:
    import pystray
    from PIL import Image, ImageDraw
    _TRAY_AVAILABLE = True
except ImportError:
    _TRAY_AVAILABLE = False
    logger.warning("pystray 或 Pillow 未安装，系统托盘不可用")


class SystemTray:
    """Windows 系统托盘"""

    def __init__(self, cfg, docker_mgr, cloud, on_quit=None):
        self.cfg = cfg
        self.docker = docker_mgr
        self.cloud = cloud
        self._on_quit = on_quit
        self._tray: "pystray.Icon" = None
        self._running = False

    def _create_icon(self, color: str) -> "Image.Image":
        """创建纯色圆形图标"""
        size = 64
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        colors = {
            "green": (0, 200, 100),
            "yellow": (255, 180, 0),
            "gray": (128, 128, 128),
            "red": (220, 50, 50),
        }
        c = colors.get(color, colors["gray"])

        draw.ellipse([8, 8, size - 8, size - 8], fill=c)
        draw.ellipse([20, 20, size - 20, size - 20], fill=(255, 255, 255, 80))

        return img

    def _refresh_icon(self):
        """根据状态刷新图标"""
        if not self._tray:
            return

        if self.cfg.status == "error":
            color = "red"
        elif self.cloud.online and self.cfg.status == "running":
            color = "green"
        elif self.cfg.status == "installing":
            color = "yellow"
        else:
            color = "gray"

        self._tray.icon = self._create_icon(color)

    def _get_menu(self):
        """构建托盘菜单"""
        if not pystray:
            return None

        gpu = self.cfg.gpu_info
        vram = f"{gpu.get('free_gb', 0)}/{gpu.get('vram_gb', 0)} GB"

        return pystray.Menu(
            pystray.MenuItem(
                f"Memento 启动器 v{self.cfg.version}",
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                f"状态: {'在线' if self.cloud.online else '离线'} | GPU: {vram}",
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "查看日志",
                self._show_logs,
            ),
            pystray.MenuItem(
                "重启容器",
                self._restart_container,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "退出",
                self._quit,
            ),
        )

    def _show_logs(self):
        """弹出日志窗口"""
        import subprocess
        import os

        log_file = Path(self.cfg.workspace).parent / "logs" / "launcher.log"
        if log_file.exists():
            if sys.platform == "win32":
                os.startfile(str(log_file))
            else:
                subprocess.Popen(["xdg-open", str(log_file)])

    def _restart_container(self):
        """重启容器"""
        logger.info("用户触发重启容器...")
        threading.Thread(target=self.docker.restart_container, daemon=True).start()

    def _quit(self):
        """退出启动器"""
        logger.info("用户触发退出...")
        if self._on_quit:
            self._on_quit()
        if self._tray:
            self._tray.stop()

    def start(self):
        """启动系统托盘"""
        if not _TRAY_AVAILABLE:
            logger.warning("系统托盘功能不可用（缺少 pystray/Pillow）")
            return

        icon = self._create_icon("gray")
        self._tray = pystray.Icon(
            "memento",
            icon,
            "Memento 启动器",
            menu=self._get_menu(),
        )

        # 定时刷新
        def refresh_loop():
            import time
            while self._running:
                time.sleep(5)
                self._refresh_icon()
                if self._tray:
                    self._tray.update_menu()

        self._running = True
        threading.Thread(target=refresh_loop, daemon=True).start()

        logger.info("系统托盘已启动")
        self._tray.run()

    def stop(self):
        """停止系统托盘"""
        self._running = False
        if self._tray:
            self._tray.stop()