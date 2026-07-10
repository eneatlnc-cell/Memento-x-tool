"""Memento 05 — 统一对齐控制图 + 四通道条件打包

输入:
  - 01 原始 30fps 帧
  - 02 Mask
  - 03 Pose 热力图
  - 04 Depth 深度图

输出:
  - Canny 轮廓图（仅在 Mask 区域内）
  - Distance 距离图（前景距离变换）
  - Pose Heatmap（已对齐尺寸/时序）
  - Temporal 时序平滑参数（相邻帧光流差异）

所有控制图统一分辨率、时序，打包为四通道条件供 06 LTX 节点使用。
"""
import logging
import json
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MementoAlign:
    """节点 5: 统一对齐 — 四通道控制条件打包

    输出四通道 RGBA:
      R: Canny 轮廓图
      G: Distance 距离图
      B: Pose 热力图
      A: Temporal 时序差分
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "heatmap_dir": ("STRING", {"default": "", "multiline": False}),
                "depth_dir": ("STRING", {"default": "", "multiline": False}),
                "canny_low": ("INT", {"default": 50, "min": 10, "max": 200, "step": 10}),
                "canny_high": ("INT", {"default": 150, "min": 50, "max": 500, "step": 10}),
                "temporal_smooth_window": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("canny_dir", "distance_dir", "pose_aligned_dir", "temporal_dir", "control_pack_dir")
    FUNCTION = "align"
    CATEGORY = "Memento/05_Align"

    def align_frame_count(self, dirs: dict, target_count: int) -> int:
        """对齐各目录帧数，取最小值"""
        counts = []
        for name, d in dirs.items():
            if d and os.path.exists(d):
                files = sorted([f for f in os.listdir(d) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                counts.append(len(files))
                logger.info(f"[MementoAlign] {name}: {len(files)} 帧")
        return min(counts) if counts else target_count

    def load_frame(self, dir_path: str, idx: int, h: int, w: int, is_gray: bool = True) -> np.ndarray:
        """加载并统一尺寸的帧"""
        files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if idx >= len(files):
            return np.zeros((h, w), dtype=np.uint8)

        flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
        img = cv2.imread(os.path.join(dir_path, files[idx]), flag)
        if img is None:
            return np.zeros((h, w), dtype=np.uint8)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        if not is_gray and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif is_gray and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def generate_canny(self, frame: np.ndarray, mask: np.ndarray | None,
                       low: int, high: int) -> np.ndarray:
        """Canny 边缘检测，仅在 Mask 区域内保留"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        edges = cv2.Canny(gray, low, high)

        if mask is not None:
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            edges = cv2.bitwise_and(edges, mask_bin)

        return edges

    def generate_distance(self, mask: np.ndarray) -> np.ndarray:
        """前景距离变换 → 距离图"""
        if mask is None:
            return np.zeros((1, 1), dtype=np.uint8)

        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        if dist.max() > 0:
            dist = (dist / dist.max() * 255).astype(np.uint8)
        return dist

    def compute_temporal_smooth(self, frames_dir: str, frame_idx: int,
                                 h: int, w: int, window: int) -> np.ndarray:
        """
        时序平滑参数：当前帧与前后帧的加权光流差异

        使用 Farneback 稠密光流计算相邻帧间运动幅度，
        取窗口内的平均运动幅度作为时序平滑权重。
        """
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if frame_idx == 0 or frame_idx >= len(frame_files) - 1:
            return np.zeros((h, w), dtype=np.uint8)

        try:
            prev = cv2.imread(os.path.join(frames_dir, frame_files[frame_idx - 1]), cv2.IMREAD_GRAYSCALE)
            curr = cv2.imread(os.path.join(frames_dir, frame_files[frame_idx]), cv2.IMREAD_GRAYSCALE)
            next_f = cv2.imread(os.path.join(frames_dir, frame_files[frame_idx + 1]), cv2.IMREAD_GRAYSCALE)

            if prev is None or curr is None or next_f is None:
                return np.zeros((h, w), dtype=np.uint8)

            prev = cv2.resize(prev, (w, h))
            curr = cv2.resize(curr, (w, h))
            next_f = cv2.resize(next_f, (w, h))

            # 前向光流
            flow_fwd = cv2.calcOpticalFlowFarneback(
                prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag_fwd = np.sqrt(flow_fwd[..., 0]**2 + flow_fwd[..., 1]**2)

            # 后向光流
            flow_bwd = cv2.calcOpticalFlowFarneback(
                next_f, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag_bwd = np.sqrt(flow_bwd[..., 0]**2 + flow_bwd[..., 1]**2)

            # 平均运动幅度
            mag_avg = (mag_fwd + mag_bwd) / 2.0

            # 归一化（运动越大 = 越需要平滑 = 值越接近 255）
            if mag_avg.max() > 0:
                mag_avg = (mag_avg / mag_avg.max() * 255).astype(np.uint8)

            return mag_avg.astype(np.uint8)

        except Exception as e:
            logger.debug(f"[MementoAlign] 时序平滑计算失败 (frame {frame_idx}): {e}")
            return np.zeros((h, w), dtype=np.uint8)

    def align(self, frames_dir: str, mask_dir: str, heatmap_dir: str,
              depth_dir: str, canny_low: int, canny_high: int, temporal_smooth_window: int):
        logger.info(
            f"[MementoAlign] 对齐控制图: frames={frames_dir}, masks={mask_dir}"
        )

        # 验证输入
        for path, name in [(frames_dir, "frames"), (mask_dir, "masks"),
                           (heatmap_dir, "heatmaps"), (depth_dir, "depth")]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"{name} 目录不存在: {path}")

        # 获取帧尺寸和帧数
        frame_files = sorted([
            f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        h, w = first.shape[:2]

        # 对齐帧数：取各目录最小值
        dirs = {"frames": frames_dir, "masks": mask_dir,
                "heatmaps": heatmap_dir, "depth": depth_dir}
        num_frames = self.align_frame_count(dirs, len(frame_files))
        logger.info(f"[MementoAlign] 对齐后帧数: {num_frames}, 尺寸: {w}x{h}")

        # 创建输出目录
        canny_dir = "/workspace/canny"
        distance_dir = "/workspace/distance"
        pose_aligned_dir = "/workspace/pose_aligned"
        temporal_dir = "/workspace/temporal"
        control_pack_dir = "/workspace/control_pack"
        for d in [canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            # 加载各通道输入
            frame = self.load_frame(frames_dir, i, h, w, is_gray=False)
            mask = self.load_frame(mask_dir, i, h, w, is_gray=True)
            heatmap = self.load_frame(heatmap_dir, i, h, w, is_gray=True)
            depth = self.load_frame(depth_dir, i, h, w, is_gray=True)

            # Canny 轮廓（仅在 Mask 内）
            canny = self.generate_canny(frame, mask, canny_low, canny_high)

            # Distance 距离图
            distance = self.generate_distance(mask)

            # Pose 热力图（已对齐，直接复制）
            pose_aligned = heatmap

            # Temporal 时序平滑
            temporal = self.compute_temporal_smooth(
                frames_dir, i, h, w, temporal_smooth_window
            )

            # 保存各通道
            cv2.imwrite(os.path.join(canny_dir, f"canny_{i+1:05d}.png"), canny)
            cv2.imwrite(os.path.join(distance_dir, f"distance_{i+1:05d}.png"), distance)
            cv2.imwrite(os.path.join(pose_aligned_dir, f"pose_{i+1:05d}.png"), pose_aligned)
            cv2.imwrite(os.path.join(temporal_dir, f"temporal_{i+1:05d}.png"), temporal)

            # 四通道打包 RGBA
            control_pack = np.stack([canny, distance, pose_aligned, temporal], axis=-1)
            cv2.imwrite(
                os.path.join(control_pack_dir, f"control_{i+1:05d}.png"),
                control_pack
            )

            if (i + 1) % 30 == 0:
                logger.info(f"[MementoAlign] 进度: {i+1}/{num_frames} 帧")

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "canny_dir": canny_dir,
            "distance_dir": distance_dir,
            "pose_aligned_dir": pose_aligned_dir,
            "temporal_dir": temporal_dir,
            "control_pack_dir": control_pack_dir,
            "num_aligned_frames": num_frames,
            "aligned_width": w,
            "aligned_height": h,
            "control_channels": ["canny", "distance", "pose", "temporal"],
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(
            f"[MementoAlign] 完成: {num_frames} 帧四通道控制图打包, "
            f"输出到 {control_pack_dir}"
        )
        return (canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir)


NODE_CLASS_MAPPINGS = {"MementoAlign": MementoAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoAlign": "Memento 05 - 统一对齐控制图"}