#!/usr/bin/env python3
"""Memento GPU Cloud CLI 启动器

用于 GPU 云实例（AutoDL、阿里云等）的无头部署。
不需要 Docker，直接管理 ComfyUI 进程。

用法:
  python launcher_cli.py --token YOUR_TOKEN
  python launcher_cli.py --token YOUR_TOKEN --comfyui /root/data/ComfyUI --port 8188

功能:
  1. 检查 GPU 环境
  2. 检查/下载模型
  3. 启动 ComfyUI 服务 (8188)
  4. 启动本地 API 服务 (8189) — 接收云端任务 + Web 管理界面
  5. 注册到云端中枢
  6. 心跳保持
  7. 优雅退出

架构:
  云端中枢 ──POST /api/v1/local/execute──▶ 本机 8189 (FastAPI) ──POST /prompt──▶ ComfyUI 8188
  云端中枢 ◀──GET  /api/v1/local/status/──  本机 8189 (FastAPI) ◀──GET /history──  ComfyUI 8188
  用户浏览器 ──HTTP────────────────────────▶ 本机 8189 (FastAPI) ── Web SPA 管理界面
"""
import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── 默认值 ──
DEFAULT_API_URL = "http://118.31.189.101:8000/api/v1"
DEFAULT_COMFYUI_DIR = "/root/data/ComfyUI"
DEFAULT_COMFYUI_PORT = 8188
DEFAULT_LOCAL_PORT = 8189
DEFAULT_MODEL_DIR = "/root/data/models"
HEARTBEAT_INTERVAL = 30

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("memento-cli")


# ═══════════════════════════════════════════════════════════
# Public IP Detection
# ═══════════════════════════════════════════════════════════

def detect_public_ip() -> str:
    """检测公网 IP（用于云端注册时填写可路由地址）"""
    services = [
        ("http://ipinfo.io/ip", 3),
        ("http://ifconfig.me/ip", 5),
        ("http://api.ipify.org", 3),
    ]
    for url, timeout in services:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                ip = resp.read().decode().strip()
                if ip:
                    logger.info(f"检测到公网 IP: {ip}")
                    return ip
        except Exception:
            continue
    # Fallback: 获取本地网络 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        logger.warning(f"无法检测公网 IP，使用本地 IP: {ip}")
        return ip
    except Exception:
        logger.warning("无法检测任何 IP，使用 127.0.0.1（云端将无法访问本机！）")
        return "127.0.0.1"


# ═══════════════════════════════════════════════════════════
# Cloud Client
# ═══════════════════════════════════════════════════════════

class CloudClient:
    """云端中枢通信客户端（纯标准库，无外部依赖）"""

    def __init__(self, api_url: str, token: str, host: str = "127.0.0.1",
                 port: int = DEFAULT_LOCAL_PORT, version: str = "2.1.0"):
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.host = host
        self.port = port
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
        """注册到云端 — 上报本机公网 IP + 本地 API 端口（8189）"""
        data = {
            "user_id": self.token,
            "host": self.host,
            "port": self.port,
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
            "host": self.host,
            "port": self.port,
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

    def __init__(self, comfyui_dir: str, port: int = DEFAULT_COMFYUI_PORT,
                 model_dir: str = DEFAULT_MODEL_DIR):
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
# Local API Server (FastAPI)
# ═══════════════════════════════════════════════════════════

class LocalServer:
    """
    本地 API 服务（端口 8189）

    提供三个核心功能：
    1. 接收云端下发的任务 → 转发到 ComfyUI 执行
    2. 托管 Web SPA 管理界面（三面板：云端/任务/模型）
    3. 健康检查端点（云端主动探活用）

    云端调用的 API 端点：
      POST /api/v1/local/execute       — 接收工作流，提交到 ComfyUI
      GET  /api/v1/local/status/{id}   — 查询任务状态
      GET  /api/v1/local/result/{id}   — 获取任务结果
      GET  /health                     — 健康检查

    用户浏览器访问：
      GET  /                           — Web 管理界面（三面板）
      GET  /api/v1/local/status        — 本地模型状态
      GET  /api/v1/local/logs          — 本地日志
    """

    def __init__(self, comfyui_port: int = DEFAULT_COMFYUI_PORT,
                 local_port: int = DEFAULT_LOCAL_PORT,
                 web_dir: str | None = None,
                 model_dir: str = DEFAULT_MODEL_DIR):
        self.comfyui_port = comfyui_port
        self.local_port = local_port
        self.web_dir = web_dir
        self.model_dir = model_dir
        self._tasks: dict = {}  # task_id → {status, prompt_id, result, error}
        self._thread: threading.Thread | None = None
        self._log_buffer: list[str] = []  # 最近 200 行日志

    def _build_app(self):
        """构建 FastAPI 应用"""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import FileResponse, PlainTextResponse
            from fastapi.staticfiles import StaticFiles
            from pydantic import BaseModel
            from contextlib import asynccontextmanager
        except ImportError as e:
            logger.error(f"缺少依赖: {e}")
            logger.error("请运行: pip install fastapi uvicorn[standard]")
            sys.exit(1)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            logger.info(f"本地 API 服务启动: 0.0.0.0:{self.local_port}")
            yield
            logger.info("本地 API 服务关闭")

        app = FastAPI(
            title="Memento Local API",
            version="2.1.0",
            lifespan=lifespan,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Pydantic 模型 ──

        class ExecuteRequest(BaseModel):
            task_id: str
            workflow: dict

        # ═══════════════════════════════════════════════════
        # 健康检查（云端主动探活用）
        # ═══════════════════════════════════════════════════

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "comfyui_port": self.comfyui_port,
                "local_port": self.local_port,
            }

        # ═══════════════════════════════════════════════════
        # 任务执行（云端 → 本机 → ComfyUI）
        # ═══════════════════════════════════════════════════

        @app.post("/api/v1/local/execute")
        async def execute_task(req: ExecuteRequest):
            """接收云端下发的工作流，转发到 ComfyUI 执行"""
            logger.info(f"收到云端任务: {req.task_id}")

            # 提交到 ComfyUI
            comfy_url = f"http://127.0.0.1:{self.comfyui_port}/prompt"
            payload = json.dumps({"prompt": req.workflow}).encode()

            try:
                comfy_req = urllib.request.Request(
                    comfy_url, data=payload,
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(comfy_req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())

                prompt_id = result.get("prompt_id")
                if not prompt_id:
                    logger.error(f"ComfyUI 未返回 prompt_id: {result}")
                    raise HTTPException(500, "ComfyUI 未返回 prompt_id")

                self._tasks[req.task_id] = {
                    "status": "running",
                    "prompt_id": prompt_id,
                    "result": None,
                    "error": None,
                }

                logger.info(f"任务已提交: {req.task_id} → prompt_id={prompt_id}")
                return {
                    "status": "accepted",
                    "task_id": req.task_id,
                    "prompt_id": prompt_id,
                }

            except urllib.error.HTTPError as e:
                logger.error(f"ComfyUI 提交失败: HTTP {e.code}")
                self._tasks[req.task_id] = {
                    "status": "failed",
                    "prompt_id": None,
                    "result": None,
                    "error": f"ComfyUI HTTP {e.code}",
                }
                raise HTTPException(502, f"ComfyUI 提交失败: HTTP {e.code}")
            except urllib.error.URLError as e:
                logger.error(f"ComfyUI 连接失败: {e.reason}")
                self._tasks[req.task_id] = {
                    "status": "failed",
                    "prompt_id": None,
                    "result": None,
                    "error": f"ComfyUI 连接失败: {e.reason}",
                }
                raise HTTPException(502, f"ComfyUI 连接失败: {e.reason}")
            except Exception as e:
                logger.error(f"任务执行异常: {e}")
                self._tasks[req.task_id] = {
                    "status": "failed",
                    "prompt_id": None,
                    "result": None,
                    "error": str(e),
                }
                raise HTTPException(500, str(e))

        @app.get("/api/v1/local/status/{task_id}")
        async def task_status(task_id: str):
            """查询任务状态（云端轮询）"""
            task = self._tasks.get(task_id)
            if not task:
                return {"task_id": task_id, "status": "not_found"}

            # 如果正在运行，查询 ComfyUI 执行状态
            if task["status"] == "running" and task.get("prompt_id"):
                self._poll_comfyui(task_id, task)

            return {
                "task_id": task_id,
                "status": task["status"],
                "error": task.get("error"),
            }

        @app.get("/api/v1/local/result/{task_id}")
        async def task_result(task_id: str):
            """获取任务结果（云端拉取）"""
            task = self._tasks.get(task_id)
            if not task:
                return {"task_id": task_id, "status": "not_found"}

            # 如果正在运行，查询 ComfyUI 执行状态
            if task["status"] == "running" and task.get("prompt_id"):
                self._poll_comfyui(task_id, task)

            return {
                "task_id": task_id,
                "status": task["status"],
                "result": task.get("result"),
                "error": task.get("error"),
            }

        # ═══════════════════════════════════════════════════
        # 本地状态（Web 管理界面用）
        # ═══════════════════════════════════════════════════

        @app.get("/api/v1/local/status")
        async def local_status():
            """本地模型和任务状态总览"""
            models = check_models(self.model_dir)
            return {
                "models": models,
                "active_tasks": len([t for t in self._tasks.values()
                                    if t["status"] == "running"]),
                "total_tasks": len(self._tasks),
                "comfyui_port": self.comfyui_port,
                "local_port": self.local_port,
            }

        @app.get("/api/v1/local/logs")
        async def local_logs(lines: int = 100):
            """获取最近日志"""
            recent = self._log_buffer[-lines:]
            return PlainTextResponse("\n".join(recent))

        # ═══════════════════════════════════════════════════
        # Web SPA 前端托管（必须在所有 API 路由之后注册）
        # ═══════════════════════════════════════════════════

        if self.web_dir and os.path.exists(os.path.join(self.web_dir, "index.html")):
            assets_dir = os.path.join(self.web_dir, "assets")
            if os.path.exists(assets_dir):
                app.mount("/assets", StaticFiles(directory=assets_dir), name="spa_assets")

            @app.get("/", include_in_schema=False)
            async def serve_index():
                return FileResponse(os.path.join(self.web_dir, "index.html"))

            @app.get("/{full_path:path}", include_in_schema=False)
            async def serve_spa(full_path: str):
                """SPA 回退：未匹配 API 路由的路径返回 index.html"""
                file_path = os.path.join(self.web_dir, full_path)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    return FileResponse(file_path)
                return FileResponse(os.path.join(self.web_dir, "index.html"))

            logger.info(f"Web 管理界面已就绪: {self.web_dir}")
        else:
            @app.get("/", include_in_schema=False)
            async def web_not_built():
                web_hint = self.web_dir or "web/dist"
                return {
                    "message": "Web 前端未构建",
                    "hint": "运行: cd web && npm install && npm run build",
                    "expected_dir": web_hint,
                }

        return app

    def _poll_comfyui(self, task_id: str, task: dict):
        """查询 ComfyUI 执行状态，更新任务状态"""
        try:
            comfy_url = f"http://127.0.0.1:{self.comfyui_port}/history/{task['prompt_id']}"
            with urllib.request.urlopen(comfy_url, timeout=5) as resp:
                history = json.loads(resp.read().decode())

            if task["prompt_id"] in history:
                h = history[task["prompt_id"]]
                if h.get("outputs"):
                    task["status"] = "completed"
                    task["result"] = h["outputs"]
                    logger.info(f"任务完成: {task_id}")
        except Exception:
            pass  # 仍在执行中

    def add_log(self, msg: str):
        """收集日志到缓冲区"""
        self._log_buffer.append(msg)
        if len(self._log_buffer) > 200:
            self._log_buffer = self._log_buffer[-200:]

    def start(self) -> bool:
        """启动本地 API 服务（后台线程）"""
        try:
            import uvicorn
        except ImportError:
            logger.error("缺少 uvicorn 依赖，请运行: pip install uvicorn[standard]")
            return False

        app = self._build_app()

        def run_server():
            uvicorn.run(app, host="0.0.0.0", port=self.local_port, log_level="warning")

        self._thread = threading.Thread(target=run_server, daemon=True, name="local-server")
        self._thread.start()

        # 等待端口就绪
        for i in range(30):
            try:
                s = socket.create_connection(("127.0.0.1", self.local_port), timeout=1)
                s.close()
                logger.info(f"本地 API 服务已就绪: 0.0.0.0:{self.local_port}")
                return True
            except Exception:
                time.sleep(0.5)

        logger.error("本地 API 服务启动超时")
        return False

    def stop(self):
        """停止本地 API 服务（daemon 线程随进程退出）"""
        pass


# ═══════════════════════════════════════════════════════════
# Main Launcher
# ═══════════════════════════════════════════════════════════

class CloudLauncher:
    """GPU 云 CLI 启动器"""

    def __init__(self, args):
        self.args = args
        self.local_port = args.local_port

        # 检测公网 IP
        if args.public_host:
            self.public_host = args.public_host
            logger.info(f"使用指定公网地址: {self.public_host}")
        else:
            self.public_host = detect_public_ip()

        self.cloud = CloudClient(
            args.api_url, args.token,
            host=self.public_host,
            port=self.local_port,
        )
        self.comfyui = ComfyUIManager(args.comfyui_dir, args.port, args.model_dir)

        # 确定 web 目录
        self.web_dir = self._resolve_web_dir()

        self.local_server = LocalServer(
            comfyui_port=args.port,
            local_port=self.local_port,
            web_dir=self.web_dir,
            model_dir=args.model_dir,
        )

        self._running = False
        self._heartbeat_thread: threading.Thread | None = None

    def _resolve_web_dir(self) -> str | None:
        """解析 web 前端构建产物目录"""
        # 1. 命令行指定
        if hasattr(self.args, 'web_dir') and self.args.web_dir:
            return self.args.web_dir

        # 2. 相对于脚本所在目录（仓库根目录）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(script_dir, "web", "dist"),
            os.path.join(script_dir, "..", "web", "dist"),
        ]
        for d in candidates:
            if os.path.exists(os.path.join(d, "index.html")):
                return d

        return None  # 未构建

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
            logger.info("运行 download_models.sh 下载缺失模型")

        # 4. 打印模型状态
        for name, info in models["models"].items():
            status = "✓" if info["ready"] else "✗"
            logger.info(f"  {status} {name}: {info['size']}")

        # 5. HTTP 服务依赖检查
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
        except ImportError:
            logger.error("缺少 FastAPI 依赖，请运行: pip install fastapi uvicorn[standard]")
            return False

        return True

    def start(self):
        """启动 ComfyUI + 本地 API 服务 + 注册 + 心跳"""
        # 1. 启动 ComfyUI
        if not self.comfyui.start():
            logger.error("ComfyUI 启动失败")
            return False

        # 2. 启动本地 API 服务 (8189)
        if not self.local_server.start():
            logger.error("本地 API 服务启动失败，云端无法下发任务！")
            logger.error("请检查端口 8189 是否被占用: lsof -i :8189")
            return False

        # 3. 注册到云端
        gpu = self.gpu_info
        if not self.cloud.register(gpu_model=gpu["model"], vram_gb=gpu["vram_gb"]):
            logger.warning("云端注册失败，将在心跳时重试")

        # 4. 启动心跳
        self._running = True
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.info("━" * 50)
        logger.info("  启动器运行中")
        logger.info(f"  ComfyUI:      http://127.0.0.1:{self.args.port}")
        logger.info(f"  本地 API:     http://0.0.0.0:{self.local_port}")
        if self.web_dir:
            logger.info(f"  Web 管理界面: http://127.0.0.1:{self.local_port}/")
        logger.info(f"  云端 API:     {self.args.api_url}")
        logger.info(f"  公网地址:     {self.public_host}:{self.local_port}")
        logger.info("  按 Ctrl+C 退出")
        logger.info("━" * 50)

        # 主循环（等待信号）
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
        self.local_server.stop()
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
  python launcher_cli.py --token YOUR_TOKEN --comfyui /root/data/ComfyUI --model-dir /root/data/models
  python launcher_cli.py --token YOUR_TOKEN --api-url http://memento.asia/api/v1
  python launcher_cli.py --token YOUR_TOKEN --public-host 123.45.67.89 --local-port 8189

架构说明:
  本机 8189 (FastAPI) ← 云端下发任务
  本机 8189 (FastAPI) → 转发到 ComfyUI 8188
  本机 8189 (FastAPI) → Web 管理界面 (如果 web/dist 已构建)
        """,
    )
    parser.add_argument("--token", required=True,
                        help="Memento 用户 Token（从 Web 端获取）")
    parser.add_argument("--api-url", default=os.getenv("MEMENTO_API_URL", DEFAULT_API_URL),
                        help=f"云端 API 地址 (默认: {DEFAULT_API_URL})")
    parser.add_argument("--comfyui", dest="comfyui_dir",
                        default=os.getenv("COMFYUI_DIR", DEFAULT_COMFYUI_DIR),
                        help=f"ComfyUI 安装目录 (默认: {DEFAULT_COMFYUI_DIR})")
    parser.add_argument("--model-dir", default=os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR),
                        help=f"模型文件目录 (默认: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("COMFYUI_PORT", str(DEFAULT_COMFYUI_PORT))),
                        help=f"ComfyUI 端口 (默认: {DEFAULT_COMFYUI_PORT})")
    parser.add_argument("--local-port", type=int,
                        default=int(os.getenv("LOCAL_PORT", str(DEFAULT_LOCAL_PORT))),
                        help=f"本地 API 服务端口 (默认: {DEFAULT_LOCAL_PORT})")
    parser.add_argument("--public-host", default=os.getenv("PUBLIC_HOST", ""),
                        help="公网 IP 或域名（云端通过此地址访问本机，默认自动检测）")
    parser.add_argument("--web-dir", default=os.getenv("WEB_DIR", ""),
                        help="Web 前端构建产物目录 (默认: 自动检测 web/dist)")
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