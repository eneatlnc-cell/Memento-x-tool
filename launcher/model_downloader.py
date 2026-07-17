"""Memento 启动器 — 模型下载管理器

负责模型自动下载、状态检查、手动下载清单。
自动下载使用国内镜像源（hf-mirror / ModelScope），单个失败不中断。
需要用户手动下载的模型（gated / 境外CDN）单独列出并给出指引。
"""
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("memento.models")

# ── 国内镜像源 ──
HF_MIRROR = "https://hf-mirror.com"
MS_ENDPOINT = "https://modelscope.cn"

# ── 默认模型目录 ──
DEFAULT_MODEL_DIR = Path(os.path.expanduser("~/.memento/workspace/models"))


# ═══════════════════════════════════════════════════════════
# 模型清单
# ═══════════════════════════════════════════════════════════

@dataclass
class ModelItem:
    """单个模型描述"""
    name: str
    category: str       # auto / manual
    size_mb: int
    size_display: str
    dir_name: str
    files: list
    reason: str
    download_url: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    ms_repo: str = ""
    hf_token_required: bool = False
    install_steps: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# 自动下载模型（启动器自动完成，国内高速镜像）
# ═══════════════════════════════════════════════════════════

AUTO_MODELS: list[ModelItem] = [
    ModelItem(
        name="MotionBERT 姿态估计",
        category="auto",
        size_mb=162,
        size_display="162 MB",
        dir_name="pose",
        files=["motionbert_ft_h36m.pth"],
        reason="",
        hf_repo="walterzhu/MotionBERT",
        download_url="https://hf-mirror.com/walterzhu/MotionBERT/resolve/main/motionbert_ft_h36m.pth",
    ),
    ModelItem(
        name="IC-LoRA Union Control",
        category="auto",
        size_mb=654,
        size_display="654 MB",
        dir_name="iclora",
        files=["ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"],
        reason="",
        hf_repo="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        hf_file="ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        download_url="https://hf-mirror.com/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    ),
    ModelItem(
        name="LTX-2.3 FP8 主模型",
        category="auto",
        size_mb=29000,
        size_display="29 GB",
        dir_name="ltx",
        files=["ltx-2.3-22b-dev-fp8.safetensors"],
        reason="",
        hf_repo="Lightricks/LTX-2.3-fp8",
        hf_file="ltx-2.3-22b-dev-fp8.safetensors",
        ms_repo="Lightricks/LTX-2.3-fp8",
        download_url="https://hf-mirror.com/Lightricks/LTX-2.3-fp8/resolve/main/ltx-2.3-22b-dev-fp8.safetensors",
    ),
    ModelItem(
        name="RAFT 光流模型",
        category="auto",
        size_mb=50,
        size_display="~50 MB",
        dir_name="raft",
        files=[],
        reason="",
        install_steps=["torchvision_raft"],
    ),
    ModelItem(
        name="MediaPipe",
        category="auto",
        size_mb=30,
        size_display="~30 MB",
        dir_name="mediapipe",
        files=[],
        reason="",
        install_steps=["pip install --no-deps mediapipe"],
    ),
    ModelItem(
        name="SAM2 源码",
        category="auto",
        size_mb=100,
        size_display="~100 MB",
        dir_name="sam2",
        files=[],
        reason="",
        install_steps=["git clone https://github.com/facebookresearch/sam2.git /opt/sam2", "pip install -e /opt/sam2"],
    ),
]

# ═══════════════════════════════════════════════════════════
# 手动下载模型（需要用户操作）
# ═══════════════════════════════════════════════════════════

MANUAL_MODELS: list[ModelItem] = [
    ModelItem(
        name="SAM2.1 权重",
        category="manual",
        size_mb=898,
        size_display="898 MB",
        dir_name="sam2",
        files=["sam2.1_hiera_large.pt"],
        reason="Meta 美国 CDN，国内下载速度慢，建议手动下载",
        download_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        hf_token_required=False,
    ),
    ModelItem(
        name="IC-LoRA Ingredients",
        category="manual",
        size_mb=1310,
        size_display="1.31 GB",
        dir_name="iclora",
        files=["ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"],
        reason="需要 HuggingFace 账号授权（gated repository）",
        download_url="https://hf-mirror.com/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
        hf_repo="Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients",
        hf_file="ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        hf_token_required=True,
    ),
    ModelItem(
        name="MediaPipe 姿态模型",
        category="manual",
        size_mb=15,
        size_display="~15 MB",
        dir_name="mediapipe",
        files=[],
        reason="首次运行从 Google 服务器下载，国内可能被墙",
        hf_token_required=False,
        install_steps=[
            "启动器运行后会自动触发下载",
            "如果失败，请设置代理或使用 VPN",
        ],
    ),
]


# ═══════════════════════════════════════════════════════════
# 模型下载管理器
# ═══════════════════════════════════════════════════════════

class ModelDownloader:
    """模型下载管理器 — 自动下载 + 手动下载指引"""

    def __init__(self, model_dir: str = None):
        self.model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
        self._ensure_dirs()
        self._progress: dict[str, dict] = {}
        self._download_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    def _ensure_dirs(self):
        """确保所有模型子目录存在"""
        for model in AUTO_MODELS + MANUAL_MODELS:
            (self.model_dir / model.dir_name).mkdir(parents=True, exist_ok=True)

    # ── 状态查询 ──

    def check_model(self, model: ModelItem) -> bool:
        """检查单个模型文件是否就绪"""
        target_dir = self.model_dir / model.dir_name
        if not target_dir.exists():
            return False
        if model.files:
            return all((target_dir / f).exists() for f in model.files)
        return any(target_dir.iterdir())

    def get_all_status(self) -> dict:
        """获取所有模型状态汇总"""
        auto_status = []
        manual_status = []
        auto_ready = True
        downloaded_gb = 0.0

        for model in AUTO_MODELS:
            ready = self.check_model(model)
            if not ready:
                auto_ready = False
            else:
                downloaded_gb += model.size_mb / 1024
            p = self._progress.get(model.name, {})
            auto_status.append({
                "name": model.name,
                "ready": ready,
                "size": model.size_display,
                "progress": p.get("progress", 0.0),
                "status": p.get("status", "ready" if ready else "pending"),
                "message": p.get("message", ""),
            })

        for model in MANUAL_MODELS:
            ready = self.check_model(model)
            if ready:
                downloaded_gb += model.size_mb / 1024
            manual_status.append({
                "name": model.name,
                "ready": ready,
                "size": model.size_display,
                "reason": model.reason,
                "dir_name": model.dir_name,
                "files": model.files,
                "download_url": model.download_url,
                "hf_token_required": model.hf_token_required,
                "hf_repo": model.hf_repo,
                "hf_file": model.hf_file,
                "install_steps": model.install_steps,
            })

        auto_total = sum(m.size_mb for m in AUTO_MODELS) / 1024
        manual_total = sum(m.size_mb for m in MANUAL_MODELS) / 1024

        return {
            "auto": auto_status,
            "manual": manual_status,
            "auto_ready": auto_ready,
            "auto_total_gb": round(auto_total, 1),
            "manual_total_gb": round(manual_total, 1),
            "downloaded_gb": round(downloaded_gb, 1),
            "model_dir": str(self.model_dir),
        }

    def get_progress(self) -> dict:
        return dict(self._progress)

    # ── 自动下载核心 ──

    def _download_wget(self, url: str, dest: str, desc: str) -> bool:
        """wget 下载（带断点续传）"""
        try:
            result = subprocess.run(
                ["wget", "-q", "--show-progress", "--continue", "-O", dest, url],
                capture_output=True, text=True, timeout=7200,
            )
            return result.returncode == 0 and os.path.exists(dest)
        except Exception as e:
            logger.warning(f"wget 失败 [{desc}]: {e}")
            return False

    def _download_hf(self, model: ModelItem) -> bool:
        """通过 huggingface_hub + hf-mirror 下载"""
        try:
            os.environ["HF_ENDPOINT"] = HF_MIRROR
            from huggingface_hub import hf_hub_download, snapshot_download

            if model.hf_file:
                hf_hub_download(
                    model.hf_repo, model.hf_file,
                    local_dir=str(self.model_dir / model.dir_name),
                    local_dir_use_symlinks=False, resume_download=True,
                )
            else:
                snapshot_download(
                    model.hf_repo,
                    local_dir=str(self.model_dir / model.dir_name),
                    local_dir_use_symlinks=False, resume_download=True,
                )
            return True
        except Exception as e:
            logger.warning(f"HF 下载失败 [{model.name}]: {e}")
            return False

    def _download_modelscope(self, model: ModelItem) -> bool:
        """通过 ModelScope 国内 CDN 下载"""
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "modelscope"],
                capture_output=True, timeout=60,
            )
            from modelscope.hub.snapshot_download import snapshot_download
            snapshot_download(
                model.ms_repo,
                local_dir=str(self.model_dir / model.dir_name),
                resume_download=True,
            )
            return True
        except Exception as e:
            logger.warning(f"ModelScope 失败 [{model.name}]: {e}")
            return False

    def _run_install_steps(self, model: ModelItem) -> bool:
        """执行特殊安装步骤"""
        try:
            for step in model.install_steps:
                if step.startswith("pip install"):
                    subprocess.run(
                        [sys.executable, "-m"] + step.split(),
                        capture_output=True, timeout=300,
                    )
                elif step.startswith("git clone"):
                    parts = step.split()
                    subprocess.run(parts, capture_output=True, timeout=300)
                elif step == "torchvision_raft":
                    subprocess.run(
                        [sys.executable, "-c",
                         "import torch, torchvision; "
                         "torchvision.models.optical_flow.raft_large("
                         "weights=torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2)"],
                        capture_output=True, timeout=120,
                    )
            return True
        except Exception as e:
            logger.warning(f"安装步骤失败 [{model.name}]: {e}")
            return False

    def _download_single(self, model: ModelItem) -> bool:
        """下载单个模型（多策略回退）"""
        self._progress[model.name] = {
            "status": "downloading", "progress": 0.0, "message": "开始下载..."
        }
        logger.info(f"  下载: {model.name} ({model.size_display})")

        success = False

        # 策略 1: wget 直链下载（最快）
        if model.download_url and model.files:
            dest = str(self.model_dir / model.dir_name / model.files[0])
            self._progress[model.name]["message"] = "wget 直链下载中..."
            success = self._download_wget(model.download_url, dest, model.name)

        # 策略 2: ModelScope（LTX 大文件优先）
        if not success and model.ms_repo:
            self._progress[model.name]["message"] = "ModelScope CDN 下载中..."
            success = self._download_modelscope(model)

        # 策略 3: hf-mirror
        if not success and model.hf_repo:
            self._progress[model.name]["message"] = "hf-mirror 下载中..."
            success = self._download_hf(model)

        # 策略 4: 特殊安装步骤
        if not success and model.install_steps:
            self._progress[model.name]["message"] = "执行安装步骤..."
            success = self._run_install_steps(model)

        if success:
            self._progress[model.name] = {
                "status": "done", "progress": 1.0, "message": "✓ 完成"
            }
            logger.info(f"  ✓ {model.name}")
        else:
            self._progress[model.name] = {
                "status": "failed", "progress": 0.0,
                "message": "✗ 下载失败，可稍后重试或手动下载"
            }
            logger.warning(f"  ✗ {model.name} 失败")

        return success

    def download_all_auto(self,
                          progress_callback: Callable = None) -> dict:
        """下载所有自动模型（同步阻塞）

        Args:
            progress_callback: (model_name, status, progress) -> None

        Returns:
            {"success": int, "failed": int, "skipped": int, "total": int}
        """
        success = 0
        failed = 0
        skipped = 0

        for model in AUTO_MODELS:
            if self._stop_flag.is_set():
                logger.info("下载已取消")
                break

            if self.check_model(model):
                self._progress[model.name] = {
                    "status": "done", "progress": 1.0, "message": "已就绪"
                }
                skipped += 1
                if progress_callback:
                    progress_callback(model.name, "done", 1.0)
                continue

            ok = self._download_single(model)
            if ok:
                success += 1
            else:
                failed += 1

            if progress_callback:
                p = self._progress.get(model.name, {})
                progress_callback(
                    model.name, p.get("status", "failed"), p.get("progress", 0.0)
                )

        return {
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "total": len(AUTO_MODELS),
        }

    def download_async(self, callback: Callable = None):
        """异步启动自动下载"""
        if self._download_thread and self._download_thread.is_alive():
            logger.warning("下载已在运行中")
            return

        self._stop_flag.clear()
        self._download_thread = threading.Thread(
            target=self.download_all_auto,
            args=(callback,),
            daemon=True,
            name="model-download",
        )
        self._download_thread.start()

    def cancel_download(self):
        """取消下载"""
        self._stop_flag.set()

    def is_downloading(self) -> bool:
        return self._download_thread is not None and self._download_thread.is_alive()

    # ── 手动下载指引 ──

    def get_manual_guide(self) -> list[dict]:
        """获取需要手动下载的模型清单及详细步骤"""
        guides = []
        for model in MANUAL_MODELS:
            if self.check_model(model):
                continue

            guide = {
                "name": model.name,
                "size": model.size_display,
                "reason": model.reason,
                "target_dir": str(self.model_dir / model.dir_name),
                "files": model.files,
                "hf_token_required": model.hf_token_required,
            }

            if model.download_url and not model.hf_token_required:
                fname = model.files[0] if model.files else "model.bin"
                guide["download_url"] = model.download_url
                guide["steps"] = [
                    f"1. 浏览器打开: {model.download_url}",
                    f"2. 下载 {fname} 到本地",
                    f"3. 将文件放入: {self.model_dir / model.dir_name / fname}",
                    f"4. 确认文件存在即完成",
                ]

            if model.hf_token_required:
                guide["download_url"] = model.download_url
                guide["steps"] = [
                    "1. 打开 https://huggingface.co/join 注册账号",
                    f"2. 打开 {model.download_url}",
                    '3. 点击 "Agree and access repository"',
                    "4. 打开 https://huggingface.co/settings/tokens",
                    '5. 创建 Read 权限 Token（格式: hf_xxxxxxxxxxxx）',
                    "6. 在启动器目录创建 hf_token.txt，粘贴 Token",
                    "7. 启动器会自动使用该 Token 从 hf-mirror 下载",
                    f"8. 或者手动下载后放入: {self.model_dir / model.dir_name}",
                ]

            if model.install_steps:
                guide["extra_notes"] = model.install_steps

            guides.append(guide)

        return guides

    def get_dir_tree(self) -> str:
        """获取模型目录 ASCII 树"""
        lines = [str(self.model_dir) + "/"]
        for model in AUTO_MODELS + MANUAL_MODELS:
            marker = "✓" if self.check_model(model) else "✗"
            tag = "[自动]" if model.category == "auto" else "[手动]"
            lines.append(f"├── {model.dir_name}/  {marker} {tag} {model.size_display}")
            for f in model.files:
                exists = (self.model_dir / model.dir_name / f).exists()
                fmarker = "✓" if exists else "✗"
                lines.append(f"│   └── {f}  {fmarker}")
        return "\n".join(lines)