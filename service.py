"""
Memento-x-tool 本地服务管理（Windows-only）

通过 memento:// 协议唤醒，管理 LocalAPIServer 的启动/停止/状态。

用法：
    python service.py --start    启动本地 API 服务
    python service.py --stop     停止本地 API 服务
    python service.py --status   查询服务状态（HTTP 健康检查）

协议：
    memento://start  →  service.py --start
    memento://stop   →  service.py --stop

启动后向云端注册，停止时注销。
"""
import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

MEMENTO_ROOT = os.path.join(os.path.expanduser("~"), ".memento")
PID_FILE = os.path.join(MEMENTO_ROOT, "server.pid")
LOG_FILE = os.path.join(MEMENTO_ROOT, "logs", "server.log")
DEFAULT_PORT = 8000
CLOUD_URL = os.environ.get("MEMENTO_CLOUD_URL", "http://localhost:8000")

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
    # 延迟添加文件日志 handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [Memento] %(levelname)s: %(message)s"))
    logger.addHandler(fh)


def _read_pid() -> int | None:
    """读取 PID 文件"""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _write_pid(pid: int):
    with open(PID_FILE, "w") as f:
        f.write(str(pid))


def _remove_pid():
    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def _is_process_running(pid: int) -> bool:
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_port_open(port: int = DEFAULT_PORT) -> bool:
    """检查本地 API 服务端口是否已监听"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _register_with_cloud():
    """向云端注册本地服务"""
    try:
        import requests
        requests.post(
            f"{CLOUD_URL}/api/v1/local/register",
            json={"host": "127.0.0.1", "port": DEFAULT_PORT},
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


# ── 命令 ──

def cmd_start():
    """启动本地 API 服务"""
    _ensure_dirs()

    # 检查是否已在运行
    existing_pid = _read_pid()
    if existing_pid and _is_process_running(existing_pid):
        if _is_port_open():
            logger.info(f"服务已在运行 (PID={existing_pid})")
            return
        else:
            logger.warning("PID 文件存在但端口未监听，清理后重启")
            _remove_pid()

    logger.info("启动 Memento-x-tool 本地 API 服务...")

    # 启动 Memento-X 本地 API 服务（子进程）
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
    """停止本地 API 服务"""
    pid = _read_pid()
    if not pid:
        logger.info("服务未在运行（无 PID 文件）")
        return

    if not _is_process_running(pid):
        logger.info(f"PID 文件存在但进程已退出 (PID={pid})")
        _remove_pid()
        return

    _unregister_from_cloud()

    logger.info(f"停止服务 (PID={pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        # 等待优雅退出
        for _ in range(10):
            time.sleep(0.5)
            if not _is_process_running(pid):
                break
        # 强制终止
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

    print(f"PID 文件:  {'存在' if pid else '无'}")
    print(f"进程存活: {'是' if running else '否'} (PID={pid})")
    print(f"端口 {DEFAULT_PORT}: {'监听中' if port_open else '未监听'}")

    if port_open:
        print("状态: 运行中 ✅")
    elif running:
        print("状态: 启动中 ⏳")
    else:
        print("状态: 已停止 ⏹")


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(description="Memento-x-tool 本地服务管理")
    parser.add_argument("--start", action="store_true", help="启动服务")
    parser.add_argument("--stop", action="store_true", help="停止服务")
    parser.add_argument("--status", action="store_true", help="查询状态")
    args = parser.parse_args()

    if args.start:
        cmd_start()
    elif args.stop:
        cmd_stop()
    elif args.status:
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()