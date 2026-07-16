"""Memento 启动器 — 本地控制服务 (FastAPI)

Web 端入口：127.0.0.1:8189
提供：Web SPA 前端 + 容器管理 + 状态查询 + 日志流式输出 + 模型下载管理
"""
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import LauncherConfig, load_config, save_config
from .docker_manager import DockerManager
from .cloud_client import CloudClient
from .model_downloader import ModelDownloader

logger = logging.getLogger("memento.server")

# ── 全局状态（由 launcher_gui.py 在启动时注入） ──
_state: dict = {
    "cfg": None,
    "docker": None,
    "cloud": None,
    "models": None,
    "install_progress": {"status": "", "progress": 0.0},
    "log_buffer": [],  # 最近 200 行日志
}


def setup_state(cfg: LauncherConfig, docker_mgr: DockerManager, cloud: CloudClient,
                models: ModelDownloader = None):
    """注入全局状态"""
    _state["cfg"] = cfg
    _state["docker"] = docker_mgr
    _state["cloud"] = cloud
    _state["models"] = models or ModelDownloader(cfg.workspace + "/models")


# ── 日志收集 ──

class LogHandler(logging.Handler):
    """将日志收集到内存缓冲区"""
    def emit(self, record):
        msg = self.format(record)
        _state["log_buffer"].append(msg)
        if len(_state["log_buffer"]) > 200:
            _state["log_buffer"] = _state["log_buffer"][-200:]


# ── Pydantic 模型 ──

class ConfigUpdate(BaseModel):
    api_url: Optional[str] = None
    user_token: Optional[str] = None
    user_id: Optional[str] = None


class StatusResponse(BaseModel):
    status: str  # idle/installing/running/error
    online: bool
    docker_running: bool
    container_running: bool
    gpu_available: bool
    gpu_model: str
    vram_gb: float
    version: str


class HealthResponse(BaseModel):
    healthy: bool
    docker: bool
    gpu: bool
    container: bool
    cloud: bool
    vram_free_gb: float
    uptime_seconds: float


# ── FastAPI 应用 ──

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("本地控制服务启动: 127.0.0.1:8189")
    yield
    logger.info("本地控制服务关闭")


app = FastAPI(
    title="Memento Launcher",
    version="2.1.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://localhost:3000",
        "http://localhost:5173",
        "https://memento-x-web.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


cfg = lambda: _state["cfg"]
docker = lambda: _state["docker"]
cloud = lambda: _state["cloud"]
models = lambda: _state["models"]


# ═══════════════════════════════════════════════════════════
# 状态端点
# ═══════════════════════════════════════════════════════════

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """获取启动器综合状态"""
    c = cfg()
    d = docker()
    gpu = c.gpu_info

    container = d.get_container()
    container_running = container is not None and container.status == "running"

    return StatusResponse(
        status=c.status,
        online=cloud().online if cloud() else False,
        docker_running=d.check_docker_running(),
        container_running=container_running,
        gpu_available=gpu.get("available", False),
        gpu_model=gpu.get("model", "unknown"),
        vram_gb=gpu.get("vram_gb", 0),
        version=c.version,
    )


@app.get("/health", response_model=HealthResponse)
async def get_health():
    """获取健康状态"""
    d = docker()
    gpu = d.get_gpu_info()
    container = d.get_container()

    return HealthResponse(
        healthy=(
            d.check_docker_running()
            and gpu.get("available", False)
            and container is not None
            and container.status == "running"
        ),
        docker=d.check_docker_running(),
        gpu=gpu.get("available", False),
        container=container is not None and container.status == "running",
        cloud=cloud().online if cloud() else False,
        vram_free_gb=gpu.get("free_gb", 0),
        uptime_seconds=time.time() - _start_time,
    )


# ═══════════════════════════════════════════════════════════
# 容器管理端点
# ═══════════════════════════════════════════════════════════

@app.post("/install")
async def install():
    """触发镜像拉取和容器启动"""
    d = docker()
    c = cfg()

    if c.status == "installing":
        raise HTTPException(409, "安装正在进行中")

    c.status = "installing"

    def on_progress(msg: str, progress: float):
        _state["install_progress"] = {"status": msg, "progress": progress}

    if not d.check_image_exists():
        logger.info("开始拉取镜像...")
        success = d.pull_image(progress_callback=on_progress)
        if not success:
            c.status = "error"
            raise HTTPException(500, "镜像拉取失败")

    logger.info("启动容器...")
    if not d.start_container():
        c.status = "error"
        raise HTTPException(500, "容器启动失败")

    time.sleep(5)
    if d.check_port():
        c.status = "running"
        logger.info("容器启动成功，8188 端口已就绪")
        return {"status": "ok", "message": "安装完成，容器已启动"}
    else:
        c.status = "error"
        raise HTTPException(500, "容器已启动但 8188 端口无响应")


@app.post("/start")
async def start_container():
    """启动已存在的容器"""
    d = docker()
    c = cfg()

    container = d.get_container()
    if container is None:
        raise HTTPException(404, "容器不存在，请先执行 /install")

    if container.status == "running":
        return {"status": "ok", "message": "容器已在运行"}

    container.start()
    time.sleep(3)
    if d.check_port():
        c.status = "running"
        return {"status": "ok", "message": "容器已启动"}
    else:
        raise HTTPException(500, "容器启动后端口无响应")


@app.post("/stop")
async def stop_container():
    """停止容器"""
    d = docker()
    c = cfg()

    if d.stop_container():
        c.status = "idle"
        return {"status": "ok", "message": "容器已停止"}
    else:
        raise HTTPException(500, "容器停止失败")


# ═══════════════════════════════════════════════════════════
# 日志端点
# ═══════════════════════════════════════════════════════════

@app.get("/logs")
async def get_logs(lines: int = Query(default=100, le=200)):
    """获取最近日志"""
    recent = _state["log_buffer"][-lines:]
    return PlainTextResponse("\n".join(recent))


@app.get("/logs/container")
async def get_container_logs(tail: int = Query(default=100, le=500)):
    """获取容器日志"""
    d = docker()
    return PlainTextResponse(d.get_logs(tail=tail))


@app.get("/logs/stream")
async def stream_logs():
    """SSE 流式日志"""
    async def generate():
        last_idx = 0
        while True:
            current = _state["log_buffer"]
            if last_idx < len(current):
                for line in current[last_idx:]:
                    yield f"data: {line}\n\n"
                last_idx = len(current)
            import asyncio
            await asyncio.sleep(0.5)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/install/progress")
async def install_progress():
    """获取安装进度"""
    return _state["install_progress"]


# ═══════════════════════════════════════════════════════════
# 配置端点
# ═══════════════════════════════════════════════════════════

@app.post("/config")
async def update_config(update: ConfigUpdate):
    """更新配置并持久化"""
    c = cfg()
    if update.api_url is not None:
        c.api_url = update.api_url
    if update.user_token is not None:
        c.user_token = update.user_token
    if update.user_id is not None:
        c.user_id = update.user_id
    save_config(c)
    return {"status": "ok", "message": "配置已更新"}


@app.get("/config")
async def get_config():
    """获取当前配置（脱敏）"""
    c = cfg()
    return {
        "api_url": c.api_url,
        "user_id": c.user_id,
        "user_token": c.user_token[:8] + "..." if c.user_token else "",
        "docker_image": c.docker_image,
        "local_port": c.local_port,
        "container_port": c.container_port,
        "version": c.version,
    }


@app.get("/gpu")
async def get_gpu_info():
    """获取 GPU 详情"""
    d = docker()
    return d.get_gpu_info()


# ═══════════════════════════════════════════════════════════
# 模型管理端点
# ═══════════════════════════════════════════════════════════

@app.get("/models/status")
async def get_models_status():
    """获取所有模型状态（自动 + 手动）"""
    m = models()
    return m.get_all_status()


@app.get("/models/progress")
async def get_models_progress():
    """获取当前下载进度"""
    m = models()
    return m.get_progress()


@app.post("/models/download")
async def start_models_download():
    """触发自动下载（异步）"""
    m = models()

    if m.is_downloading():
        raise HTTPException(409, "模型下载已在运行中")

    # 检查是否有需要下载的模型
    status = m.get_all_status()
    if status["auto_ready"]:
        return {"status": "ok", "message": "所有自动下载模型已就绪", "skipped": True}

    # 异步启动下载
    def on_progress(model_name, status, progress):
        logger.info(f"模型下载进度: {model_name} [{status}] {progress:.0%}")

    m.download_async(callback=on_progress)
    return {"status": "ok", "message": "模型下载已启动", "total": status["auto_total_gb"]}


@app.post("/models/cancel")
async def cancel_models_download():
    """取消下载"""
    m = models()
    m.cancel_download()
    return {"status": "ok", "message": "下载已取消"}


@app.get("/models/manual")
async def get_manual_guide():
    """获取手动下载模型清单及详细步骤"""
    m = models()
    return {
        "manuals": m.get_manual_guide(),
        "model_dir": str(m.model_dir),
        "dir_tree": m.get_dir_tree(),
    }


@app.get("/models/dir-tree")
async def get_models_dir_tree():
    """获取模型目录结构"""
    m = models()
    return {"dir_tree": m.get_dir_tree()}


@app.post("/models/hf-token")
async def set_hf_token(token: str = Query(...)):
    """设置 HuggingFace Token 用于下载 gated 模型"""
    if not token.startswith("hf_"):
        raise HTTPException(400, "Token 格式无效，应以 hf_ 开头")

    import os
    os.environ["HF_TOKEN"] = token

    # 写入文件持久化
    token_file = cfg().workspace + "/hf_token.txt"
    with open(token_file, "w") as f:
        f.write(token)

    logger.info("HF Token 已设置，将尝试下载 IC-LoRA Ingredients")
    return {"status": "ok", "message": "HF Token 已设置，可重新触发下载"}

# ═══════════════════════════════════════════════════════════
# Web SPA 前端托管（必须在所有 API 路由之后注册）
# ═══════════════════════════════════════════════════════════

def _get_web_dir() -> str:
    """获取 web 前端构建产物目录"""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'web', 'dist')
    else:
        return os.path.join(os.path.dirname(__file__), '..', 'web', 'dist')

_WEB_DIR = _get_web_dir()
_SPA_READY = os.path.exists(os.path.join(_WEB_DIR, 'index.html'))

if _SPA_READY:
    assets_dir = os.path.join(_WEB_DIR, 'assets')
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="spa_assets")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(os.path.join(_WEB_DIR, 'index.html'))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """SPA 回退：未匹配 API 路由的路径返回 index.html"""
        file_path = os.path.join(_WEB_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_WEB_DIR, 'index.html'))
else:
    @app.get("/", include_in_schema=False)
    async def web_not_built():
        return {"message": "Web 前端未构建，请运行: cd web && npm install && npm run build"}
