"""Memento 启动器 — 云端通信客户端

与 Memento-X 云端中枢通信：
- 注册：POST /api/v1/workflow/local/register
- 心跳：POST /api/v1/workflow/local/heartbeat（每 30s）
- 状态上报：POST /api/v1/status/report
- 注销：POST /api/v1/workflow/local/unregister
"""
import logging
import threading
import time
import json
import urllib.request
import urllib.error
from typing import Optional

from .config import LauncherConfig, HEARTBEAT_INTERVAL

logger = logging.getLogger("memento.cloud")


class CloudClient:
    """云端中枢通信客户端"""

    def __init__(self, cfg: LauncherConfig):
        self.cfg = cfg
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self.registered = False
        self.online = False

    # ── HTTP 请求 ──

    def _request(self, method: str, path: str, data: dict = None) -> Optional[dict]:
        """发送 HTTP 请求到云端 API"""
        url = f"{self.cfg.api_url}{path}"
        headers = {
            "Content-Type": "application/json",
        }
        if self.cfg.user_token:
            headers["Authorization"] = f"Bearer {self.cfg.user_token}"

        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            logger.warning(f"云端请求失败 [{method} {path}]: HTTP {e.code}")
            return None
        except urllib.error.URLError as e:
            logger.warning(f"云端连接失败 [{method} {path}]: {e.reason}")
            return None
        except Exception as e:
            logger.warning(f"云端请求异常 [{method} {path}]: {e}")
            return None

    # ── 注册/注销 ──

    def register(self) -> bool:
        """向云端注册启动器"""
        logger.info("注册到云端中枢...")

        gpu = self.cfg.gpu_info
        data = {
            "user_id": self.cfg.user_id or self.cfg.user_token,
            "host": "127.0.0.1",
            "port": self.cfg.local_port,
            "version": self.cfg.version,
            "gpu_model": gpu.get("model", "unknown"),
            "vram_gb": gpu.get("vram_gb", 0),
        }

        result = self._request("POST", "/workflow/local/register", data)
        if result and result.get("status") == "ok":
            self.registered = True
            self.online = True
            logger.info(f"注册成功: {result.get('message', '')}")
            return True
        else:
            logger.error("注册失败")
            return False

    def unregister(self) -> bool:
        """向云端注销"""
        data = {"user_id": self.cfg.user_id or self.cfg.user_token}
        result = self._request("POST", "/workflow/local/unregister", data)
        if result:
            self.registered = False
            self.online = False
            logger.info("已注销")
            return True
        return False

    def heartbeat(self) -> bool:
        """发送心跳"""
        data = {
            "user_id": self.cfg.user_id or self.cfg.user_token,
            "host": "127.0.0.1",
            "port": self.cfg.local_port,
            "version": self.cfg.version,
            "active_tasks": 0,
            "gpu_available": self.cfg.gpu_info.get("available", False),
        }
        result = self._request("POST", "/workflow/local/heartbeat", data)
        if result and result.get("status") == "ok":
            if not self.online:
                self.online = True
                logger.info("心跳恢复，状态: online")
            return True
        else:
            self.online = False
            logger.warning("心跳失败")
            return False

    def report_status(self, task_id: str, status: str, progress: float = 0.0,
                      step_id: str = "", error: str = "") -> bool:
        """上报任务状态"""
        data = {
            "task_id": task_id,
            "status": status,
            "step_id": step_id,
            "progress": progress,
            "error": error,
        }
        return self._request("POST", "/status/report", data) is not None

    def query_status(self) -> Optional[dict]:
        """查询云端状态"""
        user_id = self.cfg.user_id or self.cfg.user_token
        return self._request("GET", f"/workflow/local/status/{user_id}")

    # ── 心跳线程 ──

    def start_heartbeat(self):
        """启动心跳线程"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return

        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="heartbeat",
        )
        self._heartbeat_thread.start()
        logger.info(f"心跳线程已启动（间隔 {HEARTBEAT_INTERVAL}s）")

    def stop_heartbeat(self):
        """停止心跳线程"""
        self._stop_heartbeat.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)

    def _heartbeat_loop(self):
        """心跳循环"""
        while not self._stop_heartbeat.is_set():
            self._stop_heartbeat.wait(HEARTBEAT_INTERVAL)
            if self._stop_heartbeat.is_set():
                break
            self.heartbeat()