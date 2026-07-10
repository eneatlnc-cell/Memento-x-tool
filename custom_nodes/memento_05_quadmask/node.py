"""Memento 05 — QuadMask 四通道特征编码节点

输入: mask_dir + pose3d_json_path → 输出: 四通道特征图
四个通道: Mask / 距离场 / 姿态热图 / 时序差分
这是 Memento 自研核心编码，为节点 6 (LTX-Video) 提供控制信号
"""
import logging
import json
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MementoQuadMask:
    """节点 5: 四通道编码 — Mask + 3D 姿态 → 四通道特征

    四个通道：
    - C0: 二值掩码 (0/255)
    - C1: 距离场 (前景边缘→内部渐变)
    - C2: 姿态热图 (17 关键点高斯热图叠加)
    - C3: 时序差分 (相邻帧间变化幅度)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "pose3d_json_path": ("STRING", {"default": "", "multiline": False}),
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "gaussian_sigma": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("quadmask_dir",)
    FUNCTION = "encode"
    CATEGORY = "Memento/05_QuadMask"

    def compute_distance_field(self, mask: np.ndarray) -> np.ndarray:
        """计算掩码的距离场（欧氏距离变换）"""
        # 前景到背景边缘的距离
        dist_fg = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 背景到前景边缘的距离
        dist_bg = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)

        # 归一化到 [0, 255]
        max_dist = max(dist_fg.max(), dist_bg.max(), 1.0)
        dist_fg = (dist_fg / max_dist * 255).astype(np.uint8)

        return dist_fg

    def compute_pose_heatmap(self, pose_3d: dict, h: int, w: int, sigma: float) -> np.ndarray:
        """从 3D 关键点生成姿态热图"""
        heatmap = np.zeros((h, w), dtype=np.float32)

        for i in range(len(pose_3d["x"])):
            x = int(pose_3d["x"][i] * w)
            y = int(pose_3d["y"][i] * w)  # 注意：3D 坐标 x,y 都归一化到宽度

            # 确保在有效范围内
            if 0 <= x < w and 0 <= y < h:
                # 生成高斯核
                xs = np.arange(w, dtype=np.float32)
                ys = np.arange(h, dtype=np.float32)
                xx, yy = np.meshgrid(xs, ys)
                gauss = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
                heatmap = np.maximum(heatmap, gauss)  # 叠加取最大值

        # 归一化到 [0, 255]
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        return heatmap

    def compute_temporal_diff(self, frames_dir: str, frame_idx: int, h: int, w: int) -> np.ndarray:
        """计算时序差分（当前帧与前一帧的差异）"""
        if frame_idx == 0:
            return np.zeros((h, w), dtype=np.uint8)

        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if frame_idx >= len(frame_files):
            return np.zeros((h, w), dtype=np.uint8)

        try:
            curr = cv2.imread(os.path.join(frames_dir, frame_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            prev = cv2.imread(os.path.join(frames_dir, frame_files[frame_idx - 1]), cv2.IMREAD_GRAYSCALE)
            if curr is None or prev is None:
                return np.zeros((h, w), dtype=np.uint8)

            curr = cv2.resize(curr, (w, h))
            prev = cv2.resize(prev, (w, h))
            diff = cv2.absdiff(curr, prev)
            return diff
        except Exception:
            return np.zeros((h, w), dtype=np.uint8)

    def encode(self, mask_dir: str, pose3d_json_path: str, frames_dir: str, gaussian_sigma: float):
        logger.info(f"[MementoQuadMask] mask: {mask_dir}, pose3d: {pose3d_json_path}")

        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")
        if not os.path.exists(pose3d_json_path):
            raise FileNotFoundError(f"3D 姿态文件不存在: {pose3d_json_path}")

        # 加载姿态数据
        with open(pose3d_json_path, "r") as f:
            pose_data = json.load(f)

        # 获取掩码文件列表
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith('.png')
        ])
        if not mask_files:
            raise RuntimeError(f"掩码目录为空: {mask_dir}")

        # 创建输出目录
        quadmask_dir = "/workspace/quadmask"
        Path(quadmask_dir).mkdir(parents=True, exist_ok=True)

        # 读取第一帧掩码获取尺寸
        first_mask = cv2.imread(os.path.join(mask_dir, mask_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_mask is None:
            raise RuntimeError(f"无法读取掩码: {mask_files[0]}")
        h, w = first_mask.shape[:2]
        logger.info(f"[MementoQuadMask] {len(mask_files)} 帧, 尺寸 {w}x{h}")

        for i, mask_name in enumerate(mask_files):
            mask_path = os.path.join(mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"[MementoQuadMask] 跳过无效掩码: {mask_name}")
                continue

            # 确保尺寸一致
            mask = cv2.resize(mask, (w, h))

            # C0: 二值掩码
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            c0 = mask_bin

            # C1: 距离场
            c1 = self.compute_distance_field(mask_bin)

            # C2: 姿态热图
            frame_key = f"frame_{i+1:05d}"
            if frame_key in pose_data:
                c2 = self.compute_pose_heatmap(pose_data[frame_key], h, w, gaussian_sigma)
            else:
                c2 = np.zeros((h, w), dtype=np.uint8)

            # C3: 时序差分
            c3 = self.compute_temporal_diff(frames_dir, i, h, w)

            # 合并四通道为 RGBA PNG
            quad = np.stack([c0, c1, c2, c3], axis=-1)
            out_path = os.path.join(quadmask_dir, f"quad_{i+1:05d}.png")
            cv2.imwrite(out_path, quad)

            if (i + 1) % 30 == 0:
                logger.info(f"[MementoQuadMask] 进度: {i+1}/{len(mask_files)} 帧")

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "quadmask_dir": quadmask_dir,
            "num_quadmasks": len(mask_files),
            "quadmask_channels": ["mask", "distance_field", "pose_heatmap", "temporal_diff"],
            "quadmask_width": w,
            "quadmask_height": h,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(f"[MementoQuadMask] 完成: {len(mask_files)} 帧四通道编码, 输出到 {quadmask_dir}")
        return (quadmask_dir,)


NODE_CLASS_MAPPINGS = {"MementoQuadMask": MementoQuadMask}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoQuadMask": "Memento 05 - QuadMask 四通道编码"}