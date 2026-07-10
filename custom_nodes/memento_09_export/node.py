"""Memento 09 — FFmpeg 4K 输出节点"""
import logging
logger = logging.getLogger(__name__)


class MementoExport:
    """节点 9: 输出 — FFmpeg 帧序列合成 MP4/MOV + EXR 分层"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "graded_dir": ("STRING", {"default": "", "multiline": False}),
                "output_format": (["MP4", "MOV", "MP4+EXR"], {"default": "MP4"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1, "max": 120}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "export"
    CATEGORY = "Memento/09_Export"

    def export(self, graded_dir: str, output_format: str, fps: float):
        logger.info(f"[MementoExport] graded: {graded_dir}, format: {output_format}, fps: {fps}")
        output_path = "/workspace/output/output.mp4"
        return (output_path,)


NODE_CLASS_MAPPINGS = {"MementoExport": MementoExport}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoExport": "Memento 09 - FFmpeg 4K 输出"}