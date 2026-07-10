"""Memento 05 — QuadMask 四通道特征编码节点"""
import logging
logger = logging.getLogger(__name__)


class MementoQuadMask:
    """节点 5: 四通道编码 — Mask + 3D 姿态 → 四通道特征"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "pose3d_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("quadmask_dir",)
    FUNCTION = "encode"
    CATEGORY = "Memento/05_QuadMask"

    def encode(self, mask_dir: str, pose3d_dir: str):
        logger.info(f"[MementoQuadMask] mask: {mask_dir}, pose3d: {pose3d_dir}")
        quadmask_dir = "/workspace/quadmask"
        return (quadmask_dir,)


NODE_CLASS_MAPPINGS = {"MementoQuadMask": MementoQuadMask}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoQuadMask": "Memento 05 - QuadMask 四通道编码"}