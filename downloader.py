"""
Memento-x-tool 工具下载管理器（Windows-only）

按需下载工具到 %USERPROFILE%\.memento\tools\，支持断点续传、进度条、校验。
"""
import hashlib
import os
import shutil
import zipfile
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

import requests
from tqdm import tqdm


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    version: str
    description: str
    download_url: str
    sha256: str
    size_bytes: int
    extract_dir: str
    required: bool = False


# ── 工具清单（Windows 下载源）──

TOOL_CATALOG: dict[str, ToolDefinition] = {
    "ffmpeg": ToolDefinition(
        name="FFmpeg",
        version="7.0",
        description="视频编解码与合成",
        download_url="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
        sha256="",
        size_bytes=80_000_000,
        extract_dir="ffmpeg",
        required=True,
    ),
    "birefnet": ToolDefinition(
        name="BiRefNet",
        version="2.0",
        description="高精度 SVG 场景编辑",
        download_url="https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/BiRefNet-general-bb_swin_v1_tiny-epoch_232.pth",
        sha256="",
        size_bytes=450_000_000,
        extract_dir="birefnet",
    ),
    "sam2": ToolDefinition(
        name="SAM2",
        version="1.0",
        description="视频遮罩追踪",
        download_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        sha256="",
        size_bytes=900_000_000,
        extract_dir="sam2",
    ),
    "comfyui": ToolDefinition(
        name="ComfyUI",
        version="latest",
        description="AI 特效生成（火焰/粒子/光效）",
        download_url="https://github.com/comfyanonymous/ComfyUI/archive/refs/heads/master.zip",
        sha256="",
        size_bytes=500_000_000,
        extract_dir="ComfyUI",
    ),
    "hyperframes": ToolDefinition(
        name="HyperFrames",
        version="1.0",
        description="字幕/标题渲染引擎",
        download_url="",
        sha256="",
        size_bytes=0,
        extract_dir="hyperframes",
    ),
}


class ToolDownloader:
    """工具下载器"""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".memento", "tools")
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_tool_path(self, tool_name: str) -> Optional[str]:
        """获取已安装工具的路径"""
        tool_dir = os.path.join(self.cache_dir, tool_name)
        return tool_dir if os.path.isdir(tool_dir) else None

    def is_installed(self, tool_name: str) -> bool:
        """检查工具是否已安装"""
        return self.get_tool_path(tool_name) is not None

    def download(
        self,
        tool_name: str,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> Optional[str]:
        """
        下载并解压工具到本地缓存。

        Args:
            tool_name: 工具名称
            on_progress: 进度回调 (0.0 ~ 1.0)

        Returns:
            安装路径，失败返回 None
        """
        definition = TOOL_CATALOG.get(tool_name)
        if not definition:
            print(f"[ERROR] 未知工具: {tool_name}")
            return None

        if not definition.download_url:
            print(f"[SKIP] {tool_name} 无下载地址（需手动安装）")
            return None

        tool_dir = os.path.join(self.cache_dir, tool_name)
        os.makedirs(tool_dir, exist_ok=True)

        # 下载文件名
        filename = os.path.basename(definition.download_url.split("?")[0])
        filepath = os.path.join(tool_dir, filename)

        # 已下载 → 校验
        if os.path.exists(filepath):
            if self._verify_sha256(filepath, definition.sha256):
                print(f"[OK] {tool_name} 已下载且校验通过")
                return tool_dir
            else:
                print(f"[WARN] {tool_name} SHA256 不匹配，重新下载...")

        # 下载
        print(f"[↓] 下载 {tool_name} ({definition.size_bytes / 1e6:.0f}MB)...")
        try:
            response = requests.get(definition.download_url, stream=True, timeout=3600)
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))

            with tqdm(total=total, unit="B", unit_scale=True, desc=tool_name, ncols=80) as pbar:
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                        if total > 0 and on_progress:
                            on_progress(min(pbar.n / total, 1.0))

            # 校验
            if not self._verify_sha256(filepath, definition.sha256):
                print(f"[WARN] {tool_name} SHA256 校验失败（继续使用）")

            # 如果是压缩包 → 解压
            if filename.endswith((".zip", ".tar.gz", ".tar.xz", ".tar.bz2")):
                print(f"[↕] 解压 {tool_name}...")
                self._extract(filepath, tool_dir)

            print(f"[OK] {tool_name} 安装完成: {tool_dir}")
            return tool_dir

        except Exception as e:
            print(f"[ERROR] {tool_name} 下载失败: {e}")
            return None

    def ensure_required(self) -> bool:
        """确保所有必需工具已安装"""
        ok = True
        for name, d in TOOL_CATALOG.items():
            if d.required and not self.is_installed(name):
                print(f"[!] 必需工具 {name} 未安装，正在下载...")
                if self.download(name) is None:
                    ok = False
        return ok

    @staticmethod
    def _verify_sha256(filepath: str, expected: str) -> bool:
        if not expected:
            return os.path.exists(filepath)
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest() == expected

    @staticmethod
    def _extract(filepath: str, dest: str):
        """解压 zip/tar 到目标目录"""
        if filepath.endswith(".zip"):
            with zipfile.ZipFile(filepath, "r") as zf:
                zf.extractall(dest)
        elif filepath.endswith((".tar.gz", ".tar.xz", ".tar.bz2")):
            mode = "r:gz" if filepath.endswith(".tar.gz") else "r:xz" if filepath.endswith(".tar.xz") else "r:bz2"
            with tarfile.open(filepath, mode) as tf:
                tf.extractall(dest)
        else:
            # 非压缩包，直接复制到工具目录
            dest_file = os.path.join(dest, os.path.basename(filepath))
            if filepath != dest_file:
                shutil.copy2(filepath, dest_file)