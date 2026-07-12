"""Memento 05 — 统一对齐控制图 + 四通道条件打包 + 时序平滑

输入:
  - 01 原始 30fps 帧
  - 02 Mask (来自 SAM2.1)
  - 03 Pose 热力图 (来自 MediaPipe)
  - 04 Depth 深度图 (来自 MotionBERT)

输出:
  - Canny 轮廓图（仅在 Mask 区域内）
  - Distance 距离图（前景距离变换）
  - Pose Heatmap（已对齐尺寸/时序）
  - Temporal 时序平滑参数（相邻帧加权平均，窗口=3）
  - 四通道控制包 (RGBA: Canny/Distance/Pose/Temporal)

所有控制图统一分辨率、时序，经时序平滑后打包为四通道条件供 06 LTX 节点使用。
"""
import logging
import json
import os
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 尝试导入 memento_pipeline.ops GPU 张量操作 ──
try:
    from memento_pipeline.ops import align_controls as _ops_align_controls
    _TENSOR_OPS_AVAILABLE = True
    logger.info("[MementoAlign] memento_pipeline.ops 已加载，将使用 GPU 张量操作")
except ImportError:
    _TENSOR_OPS_AVAILABLE = False
    logger.info("[MementoAlign] memento_pipeline.ops 未安装，使用文件级回退逻辑")


class MementoAlign:
    """节点 5: 统一对齐 — 四通道控制条件打包 + 时序平滑

    输出四通道 RGBA:
      R: Canny 轮廓图
      G: Distance 距离图
      B: Pose 热力图
      A: Temporal 时序平滑（窗口=3 加权平均）
    """

    # 是否使用 GPU 张量操作
    _use_tensor_ops = _TENSOR_OPS_AVAILABLE

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

    # ------------------------------------------------------------------
    # 文件级辅助方法
    # ------------------------------------------------------------------

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

    def apply_temporal_smoothing(self, control_packs: list, window: int) -> list:
        """
        对四通道控制包应用时序平滑（滑动窗口加权平均，窗口大小=window）。

        对于每个通道的每一帧，使用前后 window//2 帧的加权平均：
          smoothed[i] = sum(weight[j] * pack[i+j]) / sum(weight[j])
        其中权重为高斯权重（中心帧权重最高）。

        Args:
            control_packs: 列表，每个元素为 (H, W, 4) uint8 四通道控制包
            window: 平滑窗口大小（奇数）

        Returns:
            平滑后的控制包列表
        """
        N = len(control_packs)
        if N < 2 or window <= 1:
            return control_packs

        half = window // 2

        # 构建高斯权重
        sigma = window / 3.0
        weights = np.exp(-0.5 * (np.arange(-half, half + 1) ** 2) / (sigma ** 2))
        weights = weights / weights.sum()

        smoothed = []
        for i in range(N):
            # 收集窗口内的帧
            window_packs = []
            window_weights = []
            for j in range(-half, half + 1):
                src_idx = max(0, min(N - 1, i + j))
                window_packs.append(control_packs[src_idx].astype(np.float32))
                window_weights.append(weights[j + half])

            # 加权平均
            ws = np.array(window_weights)
            ws_sum = ws.sum()
            if ws_sum > 0:
                ws = ws[:, None, None, None] / ws_sum

            avg = np.zeros_like(window_packs[0], dtype=np.float32)
            for k, wp in enumerate(window_packs):
                avg += wp * ws[k]

            smoothed.append(np.clip(avg, 0, 255).astype(np.uint8))

        return smoothed

    # ------------------------------------------------------------------
    # GPU 张量操作路径
    # ------------------------------------------------------------------

    def _align_tensor_ops(self, frames_dir: str, mask_dir: str, heatmap_dir: str,
                           depth_dir: str, canny_low: int, canny_high: int,
                           temporal_smooth_window: int):
        """使用 memento_pipeline.ops.align_controls 进行 GPU 张量操作"""
        if not _TENSOR_OPS_AVAILABLE:
            raise RuntimeError("memento_pipeline.ops 不可用，无法使用张量操作路径")

        import torch

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
        H, W = first.shape[:2]

        dirs = {"frames": frames_dir, "masks": mask_dir,
                "heatmaps": heatmap_dir, "depth": depth_dir}
        num_frames = self.align_frame_count(dirs, len(frame_files))
        logger.info(f"[MementoAlign] (tensor ops) 对齐后帧数: {num_frames}, 尺寸: {W}x{H}")

        # 加载所有帧为张量
        # frames: (N, 3, H, W) float32 [0, 1]
        frames_np = []
        for i in range(num_frames):
            frame = self.load_frame(frames_dir, i, H, W, is_gray=False)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_np.append(frame_rgb)
        frames_np = np.stack(frames_np, axis=0)
        frames_t = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

        # masks: (N, 1, H, W) float32 [0, 1]
        masks_np = []
        for i in range(num_frames):
            m = self.load_frame(mask_dir, i, H, W, is_gray=True)
            masks_np.append(m)
        masks_np = np.stack(masks_np, axis=0)
        masks_t = torch.from_numpy(masks_np.astype(np.float32) / 255.0).unsqueeze(1)

        # heatmaps: (N, 1, H, W) float32 [0, 1]
        heatmaps_np = []
        for i in range(num_frames):
            hm = self.load_frame(heatmap_dir, i, H, W, is_gray=True)
            heatmaps_np.append(hm)
        heatmaps_np = np.stack(heatmaps_np, axis=0)
        heatmaps_t = torch.from_numpy(heatmaps_np.astype(np.float32) / 255.0).unsqueeze(1)

        # depth_maps: (N, 1, H, W) float32 [0, 1]
        depth_np = []
        for i in range(num_frames):
            d = self.load_frame(depth_dir, i, H, W, is_gray=True)
            depth_np.append(d)
        depth_np = np.stack(depth_np, axis=0)
        depth_t = torch.from_numpy(depth_np.astype(np.float32) / 255.0).unsqueeze(1)

        # 调用 ops
        control_pack_t = _ops_align_controls(
            frames=frames_t,
            masks=masks_t,
            heatmaps=heatmaps_t,
            depth_maps=depth_t,
            canny_low=canny_low,
            canny_high=canny_high,
        )  # (N, 4, H, W) float32 [0, 1]

        # 创建输出目录
        canny_dir = "/workspace/canny"
        distance_dir = "/workspace/distance"
        pose_aligned_dir = "/workspace/pose_aligned"
        temporal_dir = "/workspace/temporal"
        control_pack_dir = "/workspace/control_pack"
        for d in [canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # 转换为 numpy 并保存
        control_pack_np = control_pack_t.cpu().numpy()  # (N, 4, H, W)
        control_packs_list = []  # (H, W, 4) uint8

        for i in range(num_frames):
            canny = (control_pack_np[i, 0] * 255).astype(np.uint8)       # R
            distance = (control_pack_np[i, 1] * 255).astype(np.uint8)    # G
            pose_aligned = (control_pack_np[i, 2] * 255).astype(np.uint8)  # B
            temporal = (control_pack_np[i, 3] * 255).astype(np.uint8)    # A

            cv2.imwrite(os.path.join(canny_dir, f"canny_{i+1:05d}.png"), canny)
            cv2.imwrite(os.path.join(distance_dir, f"distance_{i+1:05d}.png"), distance)
            cv2.imwrite(os.path.join(pose_aligned_dir, f"pose_{i+1:05d}.png"), pose_aligned)
            cv2.imwrite(os.path.join(temporal_dir, f"temporal_{i+1:05d}.png"), temporal)

            control_packs_list.append(np.stack([canny, distance, pose_aligned, temporal], axis=-1))

        # 时序平滑
        smoothed_packs = self.apply_temporal_smoothing(
            control_packs_list, temporal_smooth_window
        )

        for i in range(num_frames):
            cv2.imwrite(
                os.path.join(control_pack_dir, f"control_{i+1:05d}.png"),
                smoothed_packs[i]
            )

        if (num_frames) % 30 == 0 or num_frames <= 30:
            logger.info(f"[MementoAlign] (tensor ops) 进度: {num_frames}/{num_frames} 帧")

        # 更新 context.json
        self._update_context(canny_dir, distance_dir, pose_aligned_dir,
                             temporal_dir, control_pack_dir, num_frames, W, H)

        logger.info(
            f"[MementoAlign] (tensor ops) 完成: {num_frames} 帧四通道控制图打包, "
            f"输出到 {control_pack_dir}"
        )
        return (canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir)

    # ------------------------------------------------------------------
    # 文件级回退路径
    # ------------------------------------------------------------------

    def _align_file_based(self, frames_dir: str, mask_dir: str, heatmap_dir: str,
                           depth_dir: str, canny_low: int, canny_high: int,
                           temporal_smooth_window: int):
        """使用文件级逻辑进行控制图对齐和时序平滑"""
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
        logger.info(f"[MementoAlign] (file-based) 对齐后帧数: {num_frames}, 尺寸: {w}x{h}")

        # 创建输出目录
        canny_dir = "/workspace/canny"
        distance_dir = "/workspace/distance"
        pose_aligned_dir = "/workspace/pose_aligned"
        temporal_dir = "/workspace/temporal"
        control_pack_dir = "/workspace/control_pack"
        for d in [canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # 第一遍：逐帧计算各通道
        control_packs_list = []
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
            control_packs_list.append(control_pack)

            if (i + 1) % 30 == 0:
                logger.info(f"[MementoAlign] (file-based) 第一遍进度: {i+1}/{num_frames} 帧")

        # 第二遍：时序平滑
        logger.info(f"[MementoAlign] (file-based) 开始时序平滑 (窗口={temporal_smooth_window})...")
        smoothed_packs = self.apply_temporal_smoothing(
            control_packs_list, temporal_smooth_window
        )

        # 保存平滑后的控制包
        for i in range(num_frames):
            cv2.imwrite(
                os.path.join(control_pack_dir, f"control_{i+1:05d}.png"),
                smoothed_packs[i]
            )

        # 更新 context.json
        self._update_context(canny_dir, distance_dir, pose_aligned_dir,
                             temporal_dir, control_pack_dir, num_frames, w, h)

        logger.info(
            f"[MementoAlign] (file-based) 完成: {num_frames} 帧四通道控制图打包, "
            f"输出到 {control_pack_dir}"
        )
        return (canny_dir, distance_dir, pose_aligned_dir, temporal_dir, control_pack_dir)

    # ------------------------------------------------------------------
    # context.json 更新
    # ------------------------------------------------------------------

    def _update_context(self, canny_dir, distance_dir, pose_aligned_dir,
                        temporal_dir, control_pack_dir, num_frames, w, h):
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

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def align(self, frames_dir: str, mask_dir: str, heatmap_dir: str,
              depth_dir: str, canny_low: int, canny_high: int, temporal_smooth_window: int):
        logger.info(
            f"[MementoAlign] 对齐控制图: frames={frames_dir}, masks={mask_dir}, "
            f"tensor_ops={self._use_tensor_ops}, window={temporal_smooth_window}"
        )

        if self._use_tensor_ops and _TENSOR_OPS_AVAILABLE:
            try:
                return self._align_tensor_ops(
                    frames_dir, mask_dir, heatmap_dir, depth_dir,
                    canny_low, canny_high, temporal_smooth_window,
                )
            except Exception as e:
                logger.warning(
                    f"[MementoAlign] GPU 张量操作失败: {e}，回退到文件级逻辑"
                )

        return self._align_file_based(
            frames_dir, mask_dir, heatmap_dir, depth_dir,
            canny_low, canny_high, temporal_smooth_window,
        )


NODE_CLASS_MAPPINGS = {"MementoAlign": MementoAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoAlign": "Memento 05 - 对齐 + 时序平滑"}