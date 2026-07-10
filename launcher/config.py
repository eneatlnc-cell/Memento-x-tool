"""Memento 启动器配置管理

读取/写入 ~/.memento/config.json，持久化用户配置
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── 默认路径 ──
MEMENTO_HOME = Path(os.path.expanduser("~/.memento"))
CONFIG_PATH = MEMENTO_HOME / "config.json"
LOG_DIR = MEMENTO_HOME / "logs"
WORKSPACE_DIR = MEMENTO_HOME / "workspace"

# ── 默认值 ──
DEFAULT_API_URL = "http://118.31.189.101:8000/api/v1"
DEFAULT_IMAGE = "mementoweb/memento-tool:v1.0.0"
DEFAULT_LOCAL_PORT = 8189
DEFAULT_CONTAINER_PORT = 8188
HEARTBEAT_INTERVAL = 30


@dataclass
class LauncherConfig:
    """启动器配置"""
    api_url: str = DEFAULT_API_URL
    user_token: str = ""
    user_id: str = ""
    docker_image: str = DEFAULT_IMAGE
    local_port: int = DEFAULT_LOCAL_PORT
    container_port: int = DEFAULT_CONTAINER_PORT
    gpu_device: str = "0"
    workspace: str = str(WORKSPACE_DIR)

    # 运行态（不持久化）
    status: str = "idle"  # idle/installing/running/error
    container_id: str = ""
    gpu_info: dict = field(default_factory=lambda: {"model": "unknown", "vram_gb": 0})
    version: str = "2.0.0"


def init_dirs():
    """初始化目录结构"""
    MEMENTO_HOME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["assets", "context_buffer", "outputs"]:
        (WORKSPACE_DIR / sub).mkdir(exist_ok=True)


def load_config() -> LauncherConfig:
    """从 config.json 加载配置"""
    init_dirs()
    cfg = LauncherConfig()

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, val in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
        except (json.JSONDecodeError, Exception):
            pass

    return cfg


def save_config(cfg: LauncherConfig):
    """保存配置到 config.json（仅持久化字段）"""
    init_dirs()
    persist = {
        "api_url": cfg.api_url,
        "user_token": cfg.user_token,
        "user_id": cfg.user_id,
        "docker_image": cfg.docker_image,
        "local_port": cfg.local_port,
        "container_port": cfg.container_port,
        "gpu_device": cfg.gpu_device,
        "workspace": cfg.workspace,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(persist, f, indent=2, ensure_ascii=False)