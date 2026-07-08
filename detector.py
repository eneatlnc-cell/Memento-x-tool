"""
Memento-x-tool 硬件检测模块（Windows-only）

检测用户本地硬件能力（CPU/GPU/RAM/磁盘），决定哪些工具可用。
"""
import os
import platform
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareProfile:
    """硬件配置画像"""
    os_name: str = ""
    os_version: str = ""
    cpu_brand: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_supports_cuda: bool = False
    gpu_supports_directml: bool = False
    disk_free_gb: float = 0.0
    da_vinci_installed: bool = False

    @property
    def can_run_comfyui(self) -> bool:
        """ComfyUI 需要 >= 4GB 显存"""
        return self.gpu_vram_gb >= 4.0

    @property
    def can_run_sam2(self) -> bool:
        """SAM2 需要 >= 6GB 显存"""
        return self.gpu_vram_gb >= 6.0

    @property
    def can_run_birefnet(self) -> bool:
        """BiRefNet 可在 CPU 上运行，但 GPU 更快"""
        return True

    @property
    def recommended_toolset(self) -> list[str]:
        """根据硬件推荐可用工具集"""
        tools = ["ffmpeg", "birefnet"]
        if self.can_run_comfyui:
            tools.append("comfyui")
        if self.can_run_sam2:
            tools.append("sam2")
        if self.da_vinci_installed:
            tools.append("davinci")
        return tools

    def summary(self) -> str:
        lines = [
            f"OS:      {self.os_name} {self.os_version}",
            f"CPU:     {self.cpu_brand} ({self.cpu_cores} cores)",
            f"RAM:     {self.ram_gb:.1f} GB",
            f"GPU:     {self.gpu_name} ({self.gpu_vram_gb:.1f} GB VRAM)",
            f"CUDA:    {self.gpu_supports_cuda}",
            f"Disk:    {self.disk_free_gb:.1f} GB free",
            f"DaVinci: {self.da_vinci_installed}",
            f"Tools:   {', '.join(self.recommended_toolset)}",
        ]
        return "\n".join(lines)


class HardwareDetector:
    """Windows 硬件检测器"""

    @staticmethod
    def detect() -> HardwareProfile:
        profile = HardwareProfile()
        profile.os_name = platform.system()
        profile.os_version = platform.version()
        profile.cpu_cores = os.cpu_count() or 4

        HardwareDetector._detect_cpu(profile)
        HardwareDetector._detect_ram(profile)
        HardwareDetector._detect_gpu(profile)
        HardwareDetector._detect_disk(profile)
        HardwareDetector._detect_davinci(profile)

        return profile

    @staticmethod
    def _detect_cpu(profile: HardwareProfile):
        """通过 wmic 获取 CPU 型号"""
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
            profile.cpu_brand = lines[1] if len(lines) > 1 else platform.processor()
        except Exception:
            profile.cpu_brand = platform.processor()

    @staticmethod
    def _detect_ram(profile: HardwareProfile):
        """通过 psutil 获取内存大小"""
        try:
            import psutil
            profile.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            profile.ram_gb = 8.0

    @staticmethod
    def _detect_gpu(profile: HardwareProfile):
        """检测 GPU 型号和显存"""
        gpu_name = ""
        vram_gb = 0.0
        cuda = False

        # 1. nvidia-smi（最精确）
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                gpu_name = parts[0].strip()
                vram_gb = float(parts[1].strip()) / 1024.0
                cuda = True
                profile.gpu_name = gpu_name
                profile.gpu_vram_gb = vram_gb
                profile.gpu_supports_cuda = cuda
                return
        except Exception:
            pass

        # 2. wmic（回退）
        try:
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name,AdapterRAM"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.split("\n") if l.strip()]
            if len(lines) > 1:
                # 取第一个非标题行
                for line in lines[1:]:
                    if line and "Name" not in line:
                        parts = line.rsplit(None, 1)
                        gpu_name = line
                        if parts[-1].isdigit():
                            vram_gb = int(parts[-1]) / (1024 ** 3)
                        break
            if "NVIDIA" in gpu_name:
                cuda = True
        except Exception:
            pass

        # 3. 回退：直接检测 DirectML（Windows 11 WDDM 3.x）
        if not gpu_name:
            try:
                result = subprocess.run(
                    ["powershell", "-Command",
                     "Get-WmiObject Win32_VideoController | Select-Object Name | Format-Table -HideTableHeaders"],
                    capture_output=True, text=True, timeout=5,
                )
                gpu_name = result.stdout.strip().split("\n")[0].strip()
            except Exception:
                gpu_name = "Unknown GPU"

        profile.gpu_name = gpu_name or "Unknown GPU"
        profile.gpu_vram_gb = vram_gb
        profile.gpu_supports_cuda = cuda

    @staticmethod
    def _detect_disk(profile: HardwareProfile):
        """检测用户目录所在磁盘剩余空间"""
        try:
            import shutil
            usage = shutil.disk_usage(os.path.expanduser("~"))
            profile.disk_free_gb = usage.free / (1024 ** 3)
        except Exception:
            profile.disk_free_gb = 50.0

    @staticmethod
    def _detect_davinci(profile: HardwareProfile):
        """检查 DaVinci Resolve 是否安装"""
        prog = os.environ.get("ProgramFiles", "C:\\Program Files")
        prog86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
        paths = [
            os.path.join(prog, "Blackmagic Design", "DaVinci Resolve"),
            os.path.join(prog86, "Blackmagic Design", "DaVinci Resolve"),
        ]
        profile.da_vinci_installed = any(os.path.exists(p) for p in paths)


def detect() -> HardwareProfile:
    return HardwareDetector.detect()


if __name__ == "__main__":
    print(HardwareDetector.detect().summary())