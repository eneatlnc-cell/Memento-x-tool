"""Memento 03 — MediaPipe 2D 骨骼提取节点"""
import logging
logger = logging.getLogger(__name__)


class MementoPose2D:
    """节点 3: 2D 骨骼 — MediaPipe 33 关键点提取"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose2d_dir",)
    FUNCTION = "extract"
    CATEGORY = "Memento/03_Pose2D"

    def extract(self, frames_dir: str, mask_dir: str):
        logger.info(f"[MementoPose2D] frames: {frames_dir}, masks: {mask_dir}")
        pose2d_dir = "/workspace/pose2d"
        return (pose2d_dir,)


NODE_CLASS_MAPPINGS = {"MementoPose2D": MementoPose2D}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPose2D": "Memento 03 - MediaPipe 2D 骨骼"}