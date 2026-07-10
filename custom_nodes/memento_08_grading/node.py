"""Memento 08 — 分层光影调色节点"""
import logging
logger = logging.getLogger(__name__)


class MementoGrading:
    """节点 8: 光影调色 — 分层融合，环境光/阴影/高光匹配"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "corrected_dir": ("STRING", {"default": "", "multiline": False}),
                "original_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("graded_dir",)
    FUNCTION = "grade"
    CATEGORY = "Memento/08_Grading"

    def grade(self, corrected_dir: str, original_dir: str):
        logger.info(f"[MementoGrading] corrected: {corrected_dir}, original: {original_dir}")
        graded_dir = "/workspace/graded"
        return (graded_dir,)


NODE_CLASS_MAPPINGS = {"MementoGrading": MementoGrading}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoGrading": "Memento 08 - 分层光影调色"}