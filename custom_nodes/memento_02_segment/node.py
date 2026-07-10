"""Memento 02 — SAM3 时序分割节点

输入: 帧序列 + 用户点击坐标 → 输出: 时序一致性分割 Mask
"""
import logging

logger = logging.getLogger(__name__)


class MementoSegment:
    """节点 2: 时序分割 — SAM3-Large 像素级 Mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "points": ("STRING", {"default": "[]", "multiline": False}),
                "model": (["sam3-large",], {"default": "sam3-large"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_dir",)
    FUNCTION = "segment"
    CATEGORY = "Memento/02_Segment"

    def segment(self, frames_dir: str, points: str, model: str):
        logger.info(f"[MementoSegment] frames: {frames_dir}, model: {model}")
        # TODO: 实现 SAM3 时序分割逻辑
        mask_dir = "/workspace/masks"
        return (mask_dir,)


NODE_CLASS_MAPPINGS = {"MementoSegment": MementoSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoSegment": "Memento 02 - SAM3 时序分割"}