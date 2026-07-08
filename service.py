"""
Memento-x-tool 本地服务管理（Windows-only）

通过 memento:// 协议唤醒，管理 LocalAPIServer 的启动/停止/状态。
启动时同时启动系统托盘图标，并定期向云端发送心跳。

用法：
    python service.py --start    启动本地 API 服务 + 托盘图标
    python service.py --stop     停止本地 API 服务 + 托盘图标
    python service.py --status   查询服务状态
    python service.py --heartbeat 仅发送心跳（由定时任务调用）

协议：
    memento://start  →  service.py --start
    memento://stop   →  service.py --stop
"""
import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time

MEMENTO_ROOT = os.path.join(os.path.expanduser("~"), ".memento")
PID_FILE = os.path.join(MEMENTO_ROOT, "server.pid")
TRAY_PID_FILE = os.path.join(MEMENTO_ROOT, "tray.pid")
LOG_FILE = os.path.join(MEMENTO_ROOT, "logs", "server.log")
DEFAULT_PORT = 8000
CLOUD_URL = os.environ.get("MEMENTO_CLOUD_URL", "http://localhost:8000")
HEARTBEAT_INTERVAL = 30  # 心跳间隔（秒）

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Memento] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memento.service")


def _ensure_dirs():
    """确保目录结构存在，并初始化文件日志"""
    for d in [MEMENTO_ROOT, os.path.join(MEMENTO_ROOT, "logs"),
              os.path.join(MEMENTO_ROOT, "tools"), os.path.join(MEMENTO_ROOT, "outputs")]:
        os.makedirs(d, exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [Memento] %(levelname)s: %(message)s"))
    logger.addHandler(fh)


def _read_pid(pid_file: str = PID_FILE) -> int | None:
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _write_pid(pid: int, pid_file: str = PID_FILE):
    with open(pid_file, "w") as f:
        f.write(str(pid))


def _remove_pid(pid_file: str = PID_FILE):
    try:
        os.remove(pid_file)
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_port_open(port: int = DEFAULT_PORT) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _get_active_tasks() -> int:
    """查询本地 API 当前活跃任务数"""
    try:
        import urllib.request
        import json
        req = urllib.request.Request(f"http://127.0.0.1:{DEFAULT_PORT}/status")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return json.loads(resp.read()).get("active_tasks", 0)
    except Exception:
        return 0


def _register_with_cloud():
    """向云端注册本地服务"""
    try:
        import requests
        requests.post(
            f"{CLOUD_URL}/api/v1/local/register",
            json={"host": "127.0.0.1", "port": DEFAULT_PORT, "version": "1.0.0"},
            timeout=5,
        )
        logger.info("已向云端注册")
    except Exception as e:
        logger.warning(f"云端注册失败: {e}")


def _unregister_from_cloud():
    """从云端注销"""
    try:
        import requests
        requests.post(
            f"{CLOUD_URL}/api/v1/local/unregister",
            json={"host": "127.0.0.1", "port": DEFAULT_PORT},
            timeout=5,
        )
        logger.info("已从云端注销")
    except Exception:
        pass


def _heartbeat():
    """向云端发送心跳，附带当前状态"""
    try:
        import requests
        active_tasks = _get_active_tasks() if _is_port_open() else 0
        requests.post(
            f"{CLOUD_URL}/api/v1/local/heartbeat",
            json={
                "host": "127.0.0.1",
                "port": DEFAULT_PORT,
                "status": "running" if active_tasks > 0 else "online",
                "active_tasks": active_tasks,
            },
            timeout=5,
        )
    except Exception:
        pass  # 心跳失败不记录日志，避免刷屏


def _start_heartbeat_loop():
    """启动心跳循环（在守护线程中运行）"""
    def loop():
        while True:
            time.sleep(HEARTBEAT_INTERVAL)
            if not _is_port_open():
                # 服务已停止，退出心跳
                logger.info("端口已关闭，停止心跳")
                break
            _heartbeat()
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _start_tray():
    """启动系统托盘图标（在子进程中运行）"""
    try:
        # 检查是否已有托盘在运行
        tray_pid = _read_pid(TRAY_PID_FILE)
        if tray_pid and _is_process_running(tray_pid):
            logger.info("托盘已在运行")
            return

        tray_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tray.py")
        proc = subprocess.Popen(
            [sys.executable, tray_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        _write_pid(proc.pid, TRAY_PID_FILE)
        logger.info(f"托盘已启动 (PID={proc.pid})")
    except Exception as e:
        logger.warning(f"托盘启动失败: {e}")


def _stop_tray():
    """停止系统托盘"""
    tray_pid = _read_pid(TRAY_PID_FILE)
    if tray_pid and _is_process_running(tray_pid):
        try:
            os.kill(tray_pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.3)
                if not _is_process_running(tray_pid):
                    break
            if _is_process_running(tray_pid):
                os.kill(tray_pid, signal.SIGKILL)
        except OSError:
            pass
        _remove_pid(TRAY_PID_FILE)
        logger.info("托盘已停止")


# ── 命令 ──

def cmd_start():
    """启动本地 API 服务 + 托盘图标 + 心跳"""
    _ensure_dirs()

    # 检查是否已在运行
    existing_pid = _read_pid()
    if existing_pid and _is_process_running(existing_pid):
        if _is_port_open():
            logger.info(f"服务已在运行 (PID={existing_pid})")
            _start_tray()  # 确保托盘也在运行
            return
        else:
            logger.warning("PID 文件存在但端口未监听，清理后重启")
            _remove_pid()

    logger.info("启动 Memento-x-tool 本地 API 服务...")

    env = os.environ.copy()
    env["LOCAL_API_PORT"] = str(DEFAULT_PORT)
    env["LOCAL_API_HOST"] = "127.0.0.1"
    env["CLOUD_REGISTRY_URL"] = CLOUD_URL

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "local.api.server"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        _write_pid(proc.pid)

        # 等待端口就绪（最多 10 秒）
        for _ in range(20):
            time.sleep(0.5)
            if _is_port_open():
                logger.info(f"服务已启动 (PID={proc.pid}, port={DEFAULT_PORT})")
                _register_with_cloud()
                _start_heartbeat_loop()
                _start_tray()
                return
            if proc.poll() is not None:
                logger.error(f"服务进程异常退出 (code={proc.returncode})")
                _remove_pid()
                return

        logger.warning("服务启动超时（端口未就绪），进程可能仍在初始化")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        _remove_pid()


def cmd_stop():
    """停止本地 API 服务 + 托盘图标"""
    _unregister_from_cloud()

    _stop_tray()

    pid = _read_pid()
    if not pid:
        logger.info("服务未在运行（无 PID 文件）")
        return

    if not _is_process_running(pid):
        logger.info(f"PID 文件存在但进程已退出 (PID={pid})")
        _remove_pid()
        return

    logger.info(f"停止服务 (PID={pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            if not _is_process_running(pid):
                break
        if _is_process_running(pid):
            os.kill(pid, signal.SIGKILL)
    except OSError:
        pass

    _remove_pid()
    logger.info("服务已停止")


def cmd_status():
    """查询服务状态"""
    pid = _read_pid()
    running = pid and _is_process_running(pid)
    port_open = _is_port_open()
    active_tasks = _get_active_tasks() if port_open else 0
    tray_pid = _read_pid(TRAY_PID_FILE)
    tray_running = tray_pid and _is_process_running(tray_pid)

    print(f"PID 文件:    {'存在' if pid else '无'} (PID={pid})")
    print(f"进程存活:   {'是' if running else '否'}")
    print(f"端口 {DEFAULT_PORT}:  {'监听中' if port_open else '未监听'}")
    print(f"活跃任务:   {active_tasks}")
    print(f"托盘图标:   {'运行中' if tray_running else '未运行'}")
    print(f"心跳间隔:   {HEARTBEAT_INTERVAL}s")

    if port_open and active_tasks > 0:
        print("状态: 任务执行中 🟡")
    elif port_open:
        print("状态: 在线 🟢")
    elif running:
        print("状态: 启动中 ⏳")
    else:
        print("状态: 已停止 ⚫")


def cmd_heartbeat():
    """单次心跳发送"""
    if _is_port_open():
        _heartbeat()
        print("心跳已发送")
    else:
        print("服务未运行，跳过心跳")


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(description="Memento-x-tool 本地服务管理")
    parser.add_argument("--start", action="store_true", help="启动服务 + 托盘 + 心跳")
    parser.add_argument("--stop", action="store_true", help="停止服务 + 托盘")
    parser.add_argument("--status", action="store_true", help="查询状态")
    parser.add_argument("--heartbeat", action="store_true", help="仅发送心跳")
    args = parser.parse_args()

    if args.start:
        cmd_start()
    elif args.stop:
        cmd_stop()
    elif args.status:
        cmd_status()
    elif args.heartbeat:
        cmd_heartbeat()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()