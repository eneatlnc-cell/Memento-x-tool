"""
Memento-x-tool 系统托盘图标（Windows-only）

在系统通知区域显示 Memento 图标，提供状态指示灯和右键菜单。
依赖：pystray + Pillow

用法：
    python tray.py              # 启动托盘（自动检测服务状态）
    python tray.py --startup    # 随服务启动时调用（隐藏主窗口）
"""
import logging
import os
import sys
import threading
import time
import socket

logger = logging.getLogger("memento.tray")

MEMENTO_ROOT = os.path.join(os.path.expanduser("~"), ".memento")
PID_FILE = os.path.join(MEMENTO_ROOT, "server.pid")
DEFAULT_PORT = 8000
ICON_SIZE = 16  # 托盘图标像素


def _is_port_open(port: int = DEFAULT_PORT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _read_pid() -> int | None:
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _get_status() -> str:
    """获取当前服务状态：online / running / offline"""
    port_open = _is_port_open()
    if port_open:
        # 尝试查询是否正在执行任务
        try:
            import urllib.request
            import json
            req = urllib.request.Request(f"http://127.0.0.1:{DEFAULT_PORT}/status")
            with urllib.request.urlopen(req, timeout=1) as resp:
                data = json.loads(resp.read())
                if data.get("active_tasks", 0) > 0:
                    return "running"
        except Exception:
            pass
        return "online"
    pid = _read_pid()
    if pid and _is_process_running(pid):
        return "starting"
    return "offline"


def _create_icon(color: str) -> "PIL.Image.Image":
    """创建纯色圆形图标"""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colors = {
        "green": (0, 196, 140, 255),
        "yellow": (255, 179, 71, 255),
        "red": (255, 80, 80, 255),
        "gray": (136, 136, 136, 255),
    }
    fill = colors.get(color, colors["gray"])
    draw.ellipse([2, 2, ICON_SIZE - 2, ICON_SIZE - 2], fill=fill)
    return img


def run_tray():
    """启动系统托盘（阻塞，在线程中运行）"""
    try:
        import pystray
        from PIL import Image
    except ImportError:
        logger.warning("pystray 或 Pillow 未安装，跳过托盘图标")
        logger.warning("安装: pip install pystray Pillow")
        return

    icon = _create_icon("gray")
    tray = pystray.Icon("Memento", icon, "Memento-x-tool")

    def update_icon():
        """定时更新图标颜色"""
        last_status = ""
        while True:
            time.sleep(3)
            status = _get_status()
            if status == last_status:
                continue
            last_status = status
            color = {"online": "green", "running": "yellow", "starting": "yellow", "offline": "gray"}.get(status, "gray")
            tray.icon = _create_icon(color)
            tooltip = {"online": "Memento - 在线", "running": "Memento - 任务执行中", "starting": "Memento - 启动中", "offline": "Memento - 离线"}.get(status, "Memento")
            tray.title = tooltip

    def on_start_local():
        """菜单：启动本地服务"""
        os.system(f'pythonw.exe "{os.path.join(MEMENTO_ROOT, "launcher", "service.py")}" --start')

    def on_stop_local():
        """菜单：停止本地服务"""
        os.system(f'pythonw.exe "{os.path.join(MEMENTO_ROOT, "launcher", "service.py")}" --stop')

    def on_open_logs():
        """菜单：打开日志文件"""
        log_file = os.path.join(MEMENTO_ROOT, "logs", "server.log")
        if os.path.exists(log_file):
            os.startfile(log_file)

    def on_open_outputs():
        """菜单：打开成片输出目录"""
        outputs_dir = os.path.join(MEMENTO_ROOT, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        os.startfile(outputs_dir)

    def on_exit():
        """菜单：退出"""
        on_stop_local()
        tray.stop()

    menu = pystray.Menu(
        pystray.MenuItem("启动本地服务", on_start_local),
        pystray.MenuItem("停止本地服务", on_stop_local),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("查看日志", on_open_logs),
        pystray.MenuItem("打开成片目录", on_open_outputs),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", on_exit),
    )
    tray.menu = menu

    # 启动图标更新线程
    updater = threading.Thread(target=update_icon, daemon=True)
    updater.start()

    try:
        tray.run()
    except Exception as e:
        logger.error(f"托盘异常退出: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tray()