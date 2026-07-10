"""Memento 07 — RAFT 稠密光流节点"""
import logging
logger = logging.getLogger(__name__)


class MementoRaft:
    """节点 7: 稠密光流 — RAFT 亚像素级时序矫正"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_dir": ("STRING", {"default": "", "multiline": False}),
                "original_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("corrected_dir",)
    FUNCTION = "correct"
    CATEGORY = "Memento/07_RAFT"

    def correct(self, generated_dir: str, original_dir: str):
        logger.info(f"[MementoRaft] generated: {generated_dir}, original: {original_dir}")
        corrected_dir = "/workspace/corrected"
        return (corrected_dir,)


NODE_CLASS_MAPPINGS = {"MementoRaft": MementoRaft}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoRaft": "Memento 07 - RAFT 稠密光流"}