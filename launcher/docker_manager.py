"""Memento 启动器 — Docker 管理器

使用 docker-py 管理 ComfyUI headless 容器：
- 拉取镜像（带进度回调）
- 启动/停止/重启容器
- 检查容器状态
- 流式日志
"""
import logging
import time
from typing import Callable, Optional

import docker
from docker.errors import DockerException, ImageNotFound, NotFound, APIError

from .config import LauncherConfig

logger = logging.getLogger("memento.docker")

CONTAINER_NAME = "memento-tool"


class DockerManager:
    """Docker 容器生命周期管理"""

    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            try:
                self._client = docker.from_env()
            except DockerException as e:
                raise RuntimeError(f"Docker 不可用: {e}") from e
        return self._client

    # ── 状态检查 ──

    def check_docker_running(self) -> bool:
        """检查 Docker 守护进程是否在运行"""
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get_container(self) -> Optional[docker.models.containers.Container]:
        """获取 memento-tool 容器"""
        try:
            return self.client.containers.get(CONTAINER_NAME)
        except NotFound:
            return None
        except Exception:
            return None

    def get_status(self) -> dict:
        """获取容器状态详情"""
        container = self.get_container()
        if container is None:
            return {"status": "not_found", "running": False}

        container.reload()
        return {
            "status": container.status,
            "running": container.status == "running",
            "id": container.short_id,
            "name": container.name,
            "image": container.image.tags[0] if container.image.tags else "unknown",
            "created": container.attrs.get("Created", ""),
            "ports": container.ports,
        }

    def check_port(self) -> bool:
        """检查容器 8188 端口是否可访问"""
        import socket
        try:
            s = socket.create_connection(("127.0.0.1", self.cfg.container_port), timeout=3)
            s.close()
            return True
        except Exception:
            return False

    # ── 镜像管理 ──

    def check_image_exists(self) -> bool:
        """检查镜像是否已存在"""
        try:
            self.client.images.get(self.cfg.docker_image)
            return True
        except ImageNotFound:
            return False
        except Exception:
            return False

    def pull_image(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """拉取 Docker 镜像，可选进度回调

        Args:
            progress_callback: (status_message, progress_0_to_1) -> None

        Returns:
            True=成功
        """
        logger.info(f"拉取镜像: {self.cfg.docker_image}")
        try:
            layers = {}
            for line in self.client.api.pull(
                self.cfg.docker_image, stream=True, decode=True
            ):
                if "progress" in line:
                    layer_id = line.get("id", "")
                    progress = line.get("progressDetail", {})
                    current = progress.get("current", 0)
                    total = progress.get("total", 1)
                    layers[layer_id] = current / total if total > 0 else 0

                    # 计算总体进度
                    if layers:
                        overall = sum(layers.values()) / len(layers)
                    else:
                        overall = 0.0

                    status_msg = line.get("status", "")
                    if progress_callback:
                        progress_callback(status_msg, overall)

                elif "status" in line:
                    if progress_callback:
                        progress_callback(line["status"], 0.0)

            logger.info("镜像拉取完成")
            return True

        except APIError as e:
            logger.error(f"镜像拉取失败: {e}")
            return False

    # ── 容器管理 ──

    def start_container(self) -> bool:
        """启动容器"""

        # 先停止旧容器
        self.stop_container()

        logger.info(f"启动容器: {CONTAINER_NAME}")
        try:
            container = self.client.containers.run(
                image=self.cfg.docker_image,
                name=CONTAINER_NAME,
                detach=True,
                remove=False,
                restart_policy={"Name": "unless-stopped"},
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=[self.cfg.gpu_device],
                        capabilities=[["gpu"]],
                    )
                ] if self.cfg.gpu_device != "all" else [
                    docker.types.DeviceRequest(
                        capabilities=[["gpu"]],
                    )
                ],
                ports={f"{self.cfg.container_port}/tcp": self.cfg.container_port},
                volumes={
                    self.cfg.workspace: {"bind": "/workspace", "mode": "rw"},
                    # 模型目录单独挂载到容器内 /root/data/models（Dockerfile 约定路径）
                    # 启动器把模型下载到 ~/.memento/workspace/models，容器内通过 /root/data/models 访问
                    os.path.join(self.cfg.workspace, "models"): {"bind": "/root/data/models", "mode": "rw"},
                },
                environment={
                    "CUDA_VISIBLE_DEVICES": self.cfg.gpu_device,
                    # 告知 ComfyUI 自定义节点模型根目录
                    "COMFYUI_MODEL_DIR": "/root/data/models",
                },
            )

            self.cfg.container_id = container.short_id
            logger.info(f"容器已启动: {container.short_id}")
            return True

        except APIError as e:
            logger.error(f"容器启动失败: {e}")
            return False

    def stop_container(self) -> bool:
        """停止并删除容器"""
        container = self.get_container()
        if container is None:
            return True

        logger.info("停止容器...")
        try:
            container.stop(timeout=30)
            container.remove()
            self.cfg.container_id = ""
            logger.info("容器已停止并删除")
            return True
        except APIError as e:
            logger.error(f"停止容器失败: {e}")
            return False

    def restart_container(self) -> bool:
        """重启容器"""
        if self.stop_container():
            time.sleep(2)
            return self.start_container()
        return False

    def get_logs(self, tail: int = 100) -> str:
        """获取容器最近 N 行日志"""
        container = self.get_container()
        if container is None:
            return "容器未运行"

        try:
            logs = container.logs(tail=tail, timestamps=True)
            return logs.decode("utf-8", errors="replace")
        except APIError as e:
            return f"获取日志失败: {e}"

    # ── GPU 检测 ──

    def get_gpu_info(self) -> dict:
        """获取 GPU 信息"""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {"error": "nvidia-smi 查询失败", "available": False}

            lines = result.stdout.strip().split("\n")
            gpus = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "total_mb": int(parts[1]),
                        "used_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                    })

            if gpus:
                gpu = gpus[0]
                return {
                    "available": True,
                    "model": gpu["name"],
                    "vram_gb": round(gpu["total_mb"] / 1024, 1),
                    "used_gb": round(gpu["used_mb"] / 1024, 1),
                    "free_gb": round(gpu["free_mb"] / 1024, 1),
                    "all_gpus": gpus,
                }
            return {"available": False, "error": "未检测到 GPU"}

        except FileNotFoundError:
            return {"available": False, "error": "nvidia-smi 未安装"}
        except Exception as e:
            return {"available": False, "error": str(e)}