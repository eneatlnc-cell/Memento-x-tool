#!/usr/bin/env python3
"""Memento GPU Cloud CLI 启动器

用于 GPU 云实例（AutoDL、阿里云等）的无头部署。
不需要 Docker，直接管理 ComfyUI 进程。

用法:
  python launcher_cli.py --token YOUR_TOKEN
  python launcher_cli.py --token YOUR_TOKEN --comfyui /opt/ComfyUI --port 8188

功能:
  1. 检查 GPU 环境
  2. 检查/下载模型
  3. 启动 ComfyUI 服务
  4. 注册到云端中枢
  5. 心跳保持 + 任务监听
  6. 优雅退出
"""
import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── 默认值 ──
DEFAULT_API_URL = "http://memento.asia/api/v1"
DEFAULT_COMFYUI_DIR = "/opt/ComfyUI"
DEFAULT_PORT = 8188
DEFAULT_MODEL_DIR = "/opt/models"
HEARTBEAT_INTERVAL = 30

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memento-cli")


# ═══════════════════════════════════════════════════════════
# Cloud Client
# ═══════════════════════════════════════════════════════════

class CloudClient:
    """云端中枢通信客户端（纯标准库，无外部依赖）"""

    def __init__(self, api_url: str, token: str, version: str = "2.1.0"):
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.version = version
        self.online = False

    def _request(self, method: str, path: str, data: dict = None) -> dict | None:
        url = f"{self.api_url}{path}"
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            logger.warning(f"云端请求失败 [{method} {path}]: HTTP {e.code}")
            return None
        except Exception as e:
            logger.warning(f"云端请求失败 [{method} {path}]: {e}")
            return None

    def register(self, gpu_model: str = "", vram_gb: float = 0) -> bool:
        data = {
            "user_id": self.token,
            "host": "127.0.0.1",
            "port": DEFAULT_PORT,
            "version": self.version,
            "gpu_model": gpu_model,
            "vram_gb": vram_gb,
        }
        result = self._request("POST", "/workflow/local/register", data)
        if result and result.get("status") == "ok":
            self.online = True
            logger.info(f"云端注册成功: {result.get('message', '')}")
            return True
        return False

    def unregister(self) -> bool:
        data = {"user_id": self.token}
        result = self._request("POST", "/workflow/local/unregister", data)
        if result:
            self.online = False
            logger.info("已注销")
            return True
        return False

    def heartbeat(self) -> bool:
        data = {
            "user_id": self.token,
            "host": "127.0.0.1",
            "port": DEFAULT_PORT,
            "version": self.version,
            "active_tasks": 0,
            "gpu_available": True,
        }
        result = self._request("POST", "/workflow/local/heartbeat", data)
        if result and result.get("status") == "ok":
            if not self.online:
                self.online = True
                logger.info("心跳恢复，状态: online")
            return True
        self.online = False
        return False


# ═══════════════════════════════════════════════════════════
# GPU / Environment Check
# ═══════════════════════════════════════════════════════════

def check_gpu() -> dict:
    """检查 GPU 状态"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {"available": False, "error": "nvidia-smi 查询失败"}

        parts = [p.strip() for p in result.stdout.split(",")]
        if len(parts) >= 4:
            total_mb = int(parts[1])
            return {
                "available": True,
                "model": parts[0],
                "vram_gb": round(total_mb / 1024, 1),
                "free_gb": round(int(parts[3]) / 1024, 1),
            }
        return {"available": False, "error": "解析失败"}
    except FileNotFoundError:
        return {"available": False, "error": "nvidia-smi 未安装"}
    except Exception as e:
        return {"available": False, "error": str(e)}


def check_comfyui(comfyui_dir: str) -> bool:
    """检查 ComfyUI 是否可用"""
    main_py = Path(comfyui_dir) / "main.py"
    if not main_py.exists():
        logger.error(f"ComfyUI 未找到: {main_py}")
        return False
    return True


def check_models(model_dir: str) -> dict:
    """检查模型文件状态"""
    model_path = Path(model_dir)
    models = {
        "LTX-2.3 FP8": model_path / "ltx" / "ltx-2.3-22b-dev-fp8.safetensors",
        "IC-LoRA Union Control": model_path / "iclora" / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "IC-LoRA Ingredients": model_path / "iclora" / "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "SAM2.1": model_path / "sam2" / "sam2.1_hiera_large.pt",
        "MotionBERT": model_path / "pose" / "motionbert_ft_h36m.pth",
    }

    status = {}
    all_ready = True
    for name, path in models.items():
        ready = path.exists()
        if not ready:
            all_ready = False
        status[name] = {
            "ready": ready,
            "path": str(path),
            "size": f"{path.stat().st_size / 1024 / 1024:.0f} MB" if ready else "N/A",
        }
    return {"all_ready": all_ready, "models": status}


# ═══════════════════════════════════════════════════════════
# ComfyUI Manager
# ═══════════════════════════════════════════════════════════

class ComfyUIManager:
    """ComfyUI 进程管理器"""

    def __init__(self, comfyui_dir: str, port: int = DEFAULT_PORT, model_dir: str = DEFAULT_MODEL_DIR):
        self.comfyui_dir = comfyui_dir
        self.port = port
        self.model_dir = model_dir
        self.process: subprocess.Popen | None = None
        self._log_thread: threading.Thread | None = None

    def start(self) -> bool:
        """启动 ComfyUI 服务"""
        main_py = Path(self.comfyui_dir) / "main.py"
        if not main_py.exists():
            logger.error(f"ComfyUI main.py 不存在: {main_py}")
            return False

        env = os.environ.copy()
        env["COMFYUI_MODEL_DIR"] = self.model_dir

        cmd = [
            sys.executable, str(main_py),
            "--listen", "0.0.0.0",
            "--port", str(self.port),
        ]

        logger.info(f"启动 ComfyUI: {' '.join(cmd)}")
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.comfyui_dir,
            )

            # 后台日志线程
            def log_reader():
                for line in self.process.stdout:
                    line = line.rstrip()
                    if line:
                        logger.debug(f"[ComfyUI] {line}")

            self._log_thread = threading.Thread(target=log_reader, daemon=True)
            self._log_thread.start()

            # 等待端口就绪
            logger.info("等待 ComfyUI 启动...")
            for i in range(60):
                if self.check_ready():
                    logger.info(f"ComfyUI 已就绪 (端口 {self.port})")
                    return True
                if self.process and self.process.poll() is not None:
                    logger.error("ComfyUI 进程意外退出")
                    return False
                time.sleep(2)

            logger.error("ComfyUI 启动超时 (120s)")
            return False

        except Exception as e:
            logger.error(f"ComfyUI 启动失败: {e}")
            return False

    def check_ready(self) -> bool:
        """检查 ComfyUI 端口是否可访问"""
        import socket
        try:
            s = socket.create_connection(("127.0.0.1", self.port), timeout=3)
            s.close()
            return True
        except Exception:
            return False

    def stop(self):
        """停止 ComfyUI"""
        if self.process:
            logger.info("停止 ComfyUI...")
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
            logger.info("ComfyUI 已停止")


# ═══════════════════════════════════════════════════════════
# Main Launcher
# ═══════════════════════════════════════════════════════════

class CloudLauncher:
    """GPU 云 CLI 启动器"""

    def __init__(self, args):
        self.args = args
        self.cloud = CloudClient(args.api_url, args.token)
        self.comfyui = ComfyUIManager(args.comfyui_dir, args.port, args.model_dir)
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None

    def setup(self) -> bool:
        """环境检查"""
        logger.info("=" * 50)
        logger.info("Memento GPU Cloud Launcher v2.1")
        logger.info("=" * 50)

        # 1. GPU 检查
        gpu = check_gpu()
        if not gpu["available"]:
            logger.error(f"GPU 不可用: {gpu.get('error', '未知错误')}")
            return False
        logger.info(f"GPU: {gpu['model']} ({gpu['vram_gb']} GB, 空闲 {gpu['free_gb']} GB) ✓")
        self.gpu_info = gpu

        if gpu["vram_gb"] < 16:
            logger.error(f"显存不足: {gpu['vram_gb']} GB < 16 GB (最低要求)")
            return False

        # 2. ComfyUI 检查
        if not check_comfyui(self.args.comfyui_dir):
            logger.error(f"ComfyUI 路径: {self.args.comfyui_dir}")
            logger.info("提示: 使用 --comfyui 指定 ComfyUI 安装目录")
            return False
        logger.info(f"ComfyUI: {self.args.comfyui_dir} ✓")

        # 3. 模型检查
        models = check_models(self.args.model_dir)
        if models["all_ready"]:
            logger.info("模型: 全部就绪 ✓")
        else:
            missing = [name for name, info in models["models"].items() if not info["ready"]]
            logger.warning(f"模型: {len(missing)} 个缺失 — {', '.join(missing)}")
            logger.info(f"运行 download_models.sh 下载缺失模型")

        # 4. 打印模型状态
        for name, info in models["models"].items():
            status = "✓" if info["ready"] else "✗"
            logger.info(f"  {status} {name}: {info['size']}")

        return True

    def start(self):
        """启动 ComfyUI + 注册 + 心跳"""
        # 启动 ComfyUI
        if not self.comfyui.start():
            logger.error("ComfyUI 启动失败")
            return False

        # 注册到云端
        gpu = self.gpu_info
        if not self.cloud.register(gpu_model=gpu["model"], vram_gb=gpu["vram_gb"]):
            logger.warning("云端注册失败，将在心跳时重试")

        # 启动心跳
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.info("━" * 50)
        logger.info("  启动器运行中")
        logger.info(f"  ComfyUI: http://127.0.0.1:{self.args.port}")
        logger.info(f"  云端 API: {self.args.api_url}")
        logger.info("  按 Ctrl+C 退出")
        logger.info("━" * 50)

        # 主循环（等待信号）
        self._running = True
        while self._running:
            time.sleep(1)

        return True

    def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            time.sleep(HEARTBEAT_INTERVAL)
            if not self._running:
                break
            self.cloud.heartbeat()

    def shutdown(self):
        """优雅关闭"""
        logger.info("正在关闭...")
        self._running = False

        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

        self.cloud.unregister()
        self.comfyui.stop()
        logger.info("已关闭")


# ═══════════════════════════════════════════════════════════
# CLI Entry
# ═══════════════════════════════════════════════════════════

def get_args():
    parser = argparse.ArgumentParser(
        description="Memento GPU Cloud CLI 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python launcher_cli.py --token YOUR_TOKEN
  python launcher_cli.py --token YOUR_TOKEN --comfyui /opt/ComfyUI --model-dir /opt/models
  python launcher_cli.py --token YOUR_TOKEN --api-url http://memento.asia/api/v1
        """,
    )
    parser.add_argument("--token", required=True,
                        help="Memento 用户 Token（从 Web 端获取）")
    parser.add_argument("--api-url", default=os.getenv("MEMENTO_API_URL", DEFAULT_API_URL),
                        help=f"云端 API 地址 (默认: {DEFAULT_API_URL})")
    parser.add_argument("--comfyui", default=os.getenv("COMFYUI_DIR", DEFAULT_COMFYUI_DIR),
                        help=f"ComfyUI 安装目录 (默认: {DEFAULT_COMFYUI_DIR})")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                        help=f"模型文件目录 (默认: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--port", type=int, default=int(os.getenv("COMFYUI_PORT", str(DEFAULT_PORT))),
                        help=f"ComfyUI 端口 (默认: {DEFAULT_PORT})")
    parser.add_argument("--version", action="version", version="2.1.0")
    return parser.parse_args()


def main():
    args = get_args()

    launcher = CloudLauncher(args)

    # 信号处理
    def signal_handler(sig, frame):
        logger.info(f"收到信号 {sig}")
        launcher.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not launcher.setup():
        logger.error("环境检查未通过")
        sys.exit(1)

    try:
        launcher.start()
    except KeyboardInterrupt:
        launcher.shutdown()
    except Exception as e:
        logger.error(f"运行异常: {e}")
        launcher.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
