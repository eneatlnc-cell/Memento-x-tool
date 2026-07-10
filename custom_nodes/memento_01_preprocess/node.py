"""Memento 01 — FFmpeg 预处理节点

将 H.264/HEVC 视频拆帧为 PNG 序列，分辨率适配。
"""
import logging

logger = logging.getLogger(__name__)


class MementoPreprocess:
    """节点 1: 视频预处理 — 拆帧 + 分辨率适配"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "max_resolution": (["1080p", "4K", "8K"], {"default": "1080p"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frames_dir", "frame_count")
    FUNCTION = "process"
    CATEGORY = "Memento/01_Preprocess"

    def process(self, video_path: str, output_fps: int, max_resolution: str):
        logger.info(f"[MementoPreprocess] 输入: {video_path}, fps={output_fps}, res={max_resolution}")
        # TODO: 实现 FFmpeg 拆帧逻辑
        frames_dir = "/workspace/frames"
        frame_count = 0
        return (frames_dir, frame_count)


NODE_CLASS_MAPPINGS = {"MementoPreprocess": MementoPreprocess}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPreprocess": "Memento 01 - FFmpeg 预处理"}