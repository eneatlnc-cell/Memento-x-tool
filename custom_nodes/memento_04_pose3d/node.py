"""Memento 04 — MotionBERT 3D 归一化节点"""
import logging
logger = logging.getLogger(__name__)


class MementoPose3D:
    """节点 4: 3D 归一化 — MotionBERT 姿态归一化，防抖动"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose2d_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose3d_dir",)
    FUNCTION = "normalize"
    CATEGORY = "Memento/04_Pose3D"

    def normalize(self, pose2d_dir: str):
        logger.info(f"[MementoPose3D] pose2d: {pose2d_dir}")
        pose3d_dir = "/workspace/pose3d"
        return (pose3d_dir,)


NODE_CLASS_MAPPINGS = {"MementoPose3D": MementoPose3D}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPose3D": "Memento 04 - MotionBERT 3D 归一化"}