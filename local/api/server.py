"""
Memento 本地 API 服务器

提供本地 HTTP API，供 Memento-x-tool service.py 管理。
监听 127.0.0.1:8000，处理工作流执行和状态查询。

启动方式：
    python -m local.api.server
"""
import http.server
import json
import logging
import os
import sys
import threading
import time

logger = logging.getLogger("local.api.server")

DEFAULT_HOST = os.environ.get("LOCAL_API_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("LOCAL_API_PORT", "8000"))

# 任务状态存储
_tasks: dict = {}
_tasks_lock = threading.Lock()


class LocalAPIHandler(http.server.BaseHTTPRequestHandler):
    """本地 API 请求处理器"""

    def log_message(self, format, *args):
        logger.debug(format, *args)

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):
        if self.path == "/status":
            with _tasks_lock:
                active = sum(1 for t in _tasks.values() if t.get("status") == "running")
            self._send_json(200, {"status": "ok", "tasks": len(_tasks), "active": active})
        elif self.path.startswith("/status/"):
            task_id = self.path.split("/")[-1]
            with _tasks_lock:
                task = _tasks.get(task_id)
            if task:
                self._send_json(200, task)
            else:
                self._send_json(404, {"error": "task not found"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON"})
            return

        if self.path == "/execute":
            task_id = data.get("task_id", f"task_{int(time.time() * 1000)}")
            workflow = data.get("workflow", {})

            with _tasks_lock:
                _tasks[task_id] = {
                    "task_id": task_id,
                    "status": "running",
                    "workflow": workflow,
                    "progress": 0,
                    "steps": [],
                    "result_url": None,
                    "created_at": time.time(),
                }

            self._send_json(200, {"task_id": task_id, "status": "running"})

            # TODO: 实际工作流执行（M11-M12 渲染管线接入点）
            # 当前为占位实现，后续接入 SAM2 + MediaPipe + ComfyUI + FFmpeg 管线
            logger.info(f"任务已接收: {task_id}, workflow keys: {list(workflow.keys())}")

        elif self.path == "/cancel":
            task_id = data.get("task_id")
            with _tasks_lock:
                if task_id in _tasks:
                    _tasks[task_id]["status"] = "cancelled"
                    self._send_json(200, {"task_id": task_id, "status": "cancelled"})
                else:
                    self._send_json(404, {"error": "task not found"})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    server = http.server.HTTPServer((DEFAULT_HOST, DEFAULT_PORT), LocalAPIHandler)
    logger.info(f"本地 API 服务已启动: http://{DEFAULT_HOST}:{DEFAULT_PORT}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
        server.shutdown()


if __name__ == "__main__":
    main()