"""Memento 06 — Wan3-DiT + VACE3 影视级重绘节点"""
import logging
logger = logging.getLogger(__name__)


class MementoWan3:
    """节点 6: 影视级重绘 — Wan3-DiT + VACE3 身份锁定合成"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "quadmask_dir": ("STRING", {"default": "", "multiline": False}),
                "pose3d_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_dir",)
    FUNCTION = "generate"
    CATEGORY = "Memento/06_Wan3"

    def generate(self, frames_dir: str, quadmask_dir: str, pose3d_dir: str):
        logger.info(f"[MementoWan3] frames: {frames_dir}, quadmask: {quadmask_dir}")
        generated_dir = "/workspace/generated"
        return (generated_dir,)


NODE_CLASS_MAPPINGS = {"MementoWan3": MementoWan3}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoWan3": "Memento 06 - Wan3-DiT+VACE3 影视级重绘"}