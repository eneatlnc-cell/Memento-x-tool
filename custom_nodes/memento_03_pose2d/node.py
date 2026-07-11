"""Memento 03 — MediaPipe 33点 2D人体姿态检测 + 骨骼热力图

输入:
  - 01 原始 30fps 帧 (frames_dir)
  - 02 人物 Mask (mask_dir) — 来自 SAM3 视频分割

输出:
  - Pose 骨骼热力图（每帧一张热力图 PNG）
  - 关键点 JSON（33 关键点坐标 + 可见度）

结合 Mask 过滤干扰:
  - Mask 外区域涂黑后检测 → 消除背景人物干扰
  - 关键点落在 Mask 外 → visibility=0
  - 骨骼热力图仅在 Mask 区域内绘制

控制信号来源:
  - Mask  → 来自 02 SAM3 视频分割
  - 原始帧 → 来自 01 原始 30fps 帧
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
    from memento_pipeline.ops import extract_pose_2d as _ops_extract_pose_2d
    _TENSOR_OPS_AVAILABLE = True
    logger.info("[MementoPose2D] memento_pipeline.ops 已加载，将使用 GPU 张量操作")
except ImportError:
    _TENSOR_OPS_AVAILABLE = False
    logger.info("[MementoPose2D] memento_pipeline.ops 未安装，使用文件级回退逻辑")

# ── MediaPipe 导入 ──
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("[MementoPose2D] mediapipe 未安装，将使用空关键点")

# ── MediaPipe 33 关键点间的骨架连线 ──
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),    # 鼻→左眼→左眼内→左耳
    (0, 4), (4, 5), (5, 6), (6, 8),    # 鼻→右眼→右眼外→右耳
    (9, 10),                             # 嘴
    (11, 12), (11, 13), (13, 15),       # 左肩→左肘→左腕
    (12, 14), (14, 16),                  # 右肩→右肘→右腕
    (11, 23), (12, 24), (23, 24),       # 肩→髋
    (23, 25), (25, 27), (27, 29), (29, 31),  # 左腿
    (24, 26), (26, 28), (28, 30), (30, 32),  # 右腿
]


class MementoPose2D:
    """节点 3: 2D 骨骼 — MediaPipe 33 关键点 + 骨骼热力图 (Mask 过滤)

    输入: 01 原始帧 + 02 人物 Mask
    输出: 骨骼热力图 + 关键点 JSON
    """

    # 是否使用 GPU 张量操作
    _use_tensor_ops = _TENSOR_OPS_AVAILABLE

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "model_complexity": ("INT", {
                    "default": 2, "min": 0, "max": 2, "step": 1,
                    "tooltip": "0=轻量(最快), 1=平衡, 2=完整(最准)",
                }),
                "min_detection_confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "检测最小置信度，远距离/遮挡可降至 0.3",
                }),
                "min_tracking_confidence": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "追踪最小置信度，提高可增加鲁棒性",
                }),
                "heatmap_sigma": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 20.0, "step": 1.0,
                    "tooltip": "热力图模糊半径，越小控制越严格",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pose_json_path", "heatmap_dir")
    FUNCTION = "extract"
    CATEGORY = "Memento/03_Pose2D"

    # ------------------------------------------------------------------
    # 文件级辅助方法（回退逻辑使用）
    # ------------------------------------------------------------------

    def load_mask(self, mask_path: str, target_h: int, target_w: int) -> np.ndarray | None:
        if not os.path.exists(mask_path):
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h))
        return mask

    def keypoints_to_dict(self, landmarks, frame_w: int, frame_h: int) -> dict:
        kp = {"x": [], "y": [], "z": [], "visibility": []}
        for lm in landmarks.landmark:
            kp["x"].append(round(lm.x, 6))
            kp["y"].append(round(lm.y, 6))
            kp["z"].append(round(lm.z, 6))
            kp["visibility"].append(round(lm.visibility, 6))
        return kp

    def generate_pose_heatmap(self, keypoints: dict, mask: np.ndarray | None,
                               h: int, w: int, sigma: float) -> np.ndarray:
        """
        从关键点生成骨骼热力图（仅在 Mask 区域内绘制）

        - 对每个可见关键点，画高斯热斑
        - 对每条骨架连线，画线段高斯热斑
        - 最终用 Mask 裁切，Mask 外置零
        """
        heatmap = np.zeros((h, w), dtype=np.float32)

        xs = keypoints["x"]
        ys = keypoints["y"]
        vis = keypoints["visibility"]

        # 高斯核生成辅助函数
        def add_gaussian(x: float, y: float, weight: float = 1.0):
            px = int(x * w)
            py = int(y * h)
            if px < 0 or px >= w or py < 0 or py >= h:
                return
            ys_grid, xs_grid = np.ogrid[:h, :w]
            gauss = np.exp(-((xs_grid - px)**2 + (ys_grid - py)**2) / (2 * sigma**2))
            np.maximum(heatmap, gauss * weight, out=heatmap)

        # 绘制关键点热斑
        for i in range(len(xs)):
            if vis[i] > 0.3:
                add_gaussian(xs[i], ys[i], vis[i])

        # 绘制骨架连线热斑（沿线段采样）
        for conn in SKELETON_CONNECTIONS:
            i, j = conn
            if i < len(xs) and j < len(xs) and vis[i] > 0.3 and vis[j] > 0.3:
                for t in np.linspace(0, 1, 20):
                    px = xs[i] * (1 - t) + xs[j] * t
                    py = ys[i] * (1 - t) + ys[j] * t
                    add_gaussian(px, py, 0.5)

        # Mask 裁切
        if mask is not None:
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            heatmap[mask_bin == 0] = 0

        # 归一化到 [0, 255]
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        return heatmap.astype(np.uint8)

    # ------------------------------------------------------------------
    # GPU 张量操作路径
    # ------------------------------------------------------------------

    def _extract_tensor_ops(self, frames_dir: str, mask_dir: str,
                             min_detection_confidence: float,
                             min_tracking_confidence: float,
                             heatmap_sigma: float):
        """使用 memento_pipeline.ops.extract_pose_2d 进行 GPU 张量操作"""
        if not _TENSOR_OPS_AVAILABLE:
            raise RuntimeError("memento_pipeline.ops 不可用，无法使用张量操作路径")

        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        # 加载所有帧为张量 (N, 3, H, W) float32 [0, 1]
        import torch
        frames_np = []
        for fn in frame_files:
            img = cv2.imread(os.path.join(frames_dir, fn))
            if img is None:
                raise RuntimeError(f"无法读取帧: {fn}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_np.append(img)
        frames_np = np.stack(frames_np, axis=0)  # (N, H, W, 3)
        N, H, W, _ = frames_np.shape
        frames_t = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # (N, 3, H, W)

        # 加载所有掩码为张量 (N, 1, H, W) float32 [0, 1]
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith('.png')
        ])
        masks_np = np.zeros((N, H, W), dtype=np.float32)
        for i in range(min(N, len(mask_files))):
            mask_path = os.path.join(mask_dir, mask_files[i])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != (H, W):
                    mask = cv2.resize(mask, (W, H))
                masks_np[i] = mask.astype(np.float32) / 255.0
        masks_t = torch.from_numpy(masks_np).unsqueeze(1)  # (N, 1, H, W)

        # 调用 ops
        keypoints_dict, heatmaps_t = _ops_extract_pose_2d(
            frames=frames_t,
            masks=masks_t,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # 创建输出目录
        pose_dir = "/workspace/pose"
        heatmap_dir = "/workspace/pose_heatmap"
        Path(pose_dir).mkdir(parents=True, exist_ok=True)
        Path(heatmap_dir).mkdir(parents=True, exist_ok=True)

        # 保存热力图
        heatmaps_np_out = heatmaps_t.cpu().squeeze(1).numpy()  # (N, H, W)
        for i in range(N):
            hm = heatmaps_np_out[i]
            if hm.max() > 0:
                hm = (hm / hm.max() * 255).astype(np.uint8)
            else:
                hm = np.zeros((H, W), dtype=np.uint8)
            heatmap_path = os.path.join(heatmap_dir, f"heatmap_{i+1:05d}.png")
            cv2.imwrite(heatmap_path, hm)

        # 保存 JSON
        pose_json_path = os.path.join(pose_dir, "keypoints.json")
        with open(pose_json_path, "w") as f:
            json.dump(keypoints_dict, f, indent=2)

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "pose_json_path": pose_json_path,
            "heatmap_dir": heatmap_dir,
            "num_pose_frames": len(keypoints_dict),
            "keypoint_count": 33,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        detected_frames = sum(
            1 for v in keypoints_dict.values() if v["visibility"][0] > 0
        )
        logger.info(
            f"[MementoPose2D] (tensor ops) 完成: {detected_frames}/{len(keypoints_dict)} 帧检测到姿势, "
            f"热力图输出到 {heatmap_dir}"
        )
        return (pose_json_path, heatmap_dir)

    # ------------------------------------------------------------------
    # 文件级回退路径
    # ------------------------------------------------------------------

    def _extract_file_based(self, frames_dir: str, mask_dir: str,
                             min_detection_confidence: float,
                             min_tracking_confidence: float,
                             heatmap_sigma: float):
        """使用 MediaPipe + 文件级逻辑进行姿态检测"""
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe 未安装，且 memento_pipeline.ops 不可用。无法执行姿态检测。")

        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"帧目录不存在: {frames_dir}")

        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        # 创建输出目录
        pose_dir = "/workspace/pose"
        heatmap_dir = "/workspace/pose_heatmap"
        Path(pose_dir).mkdir(parents=True, exist_ok=True)
        Path(heatmap_dir).mkdir(parents=True, exist_ok=True)

        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        h, w = first_frame.shape[:2]
        logger.info(f"[MementoPose2D] {len(frame_files)} 帧, 尺寸 {w}x{h}")

        all_keypoints = {}

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as pose:

            for i, frame_name in enumerate(frame_files):
                frame_path = os.path.join(frames_dir, frame_name)
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"[MementoPose2D] 无法读取帧: {frame_path}")
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 加载对应掩码
                mask_name = f"mask_{i+1:05d}.png"
                mask_path = os.path.join(mask_dir, mask_name)
                mask = self.load_mask(mask_path, h, w)

                # Mask 区域外涂黑 → 消除背景人物干扰
                if mask is not None:
                    masked_frame = frame_rgb.copy()
                    masked_frame[mask == 0] = 0
                    results = pose.process(masked_frame)
                else:
                    results = pose.process(frame_rgb)

                frame_key = f"frame_{i+1:05d}"

                if results.pose_landmarks:
                    valid_kp = self.keypoints_to_dict(results.pose_landmarks, w, h)

                    # 过滤 Mask 外的关键点
                    if mask is not None:
                        for j in range(len(valid_kp["x"])):
                            px = int(valid_kp["x"][j] * w)
                            py = int(valid_kp["y"][j] * h)
                            if 0 <= px < w and 0 <= py < h:
                                if mask[py, px] == 0:
                                    valid_kp["visibility"][j] = 0.0

                    all_keypoints[frame_key] = valid_kp

                    # 生成骨骼热力图
                    heatmap = self.generate_pose_heatmap(valid_kp, mask, h, w, heatmap_sigma)
                else:
                    all_keypoints[frame_key] = {
                        "x": [0.0] * 33, "y": [0.0] * 33,
                        "z": [0.0] * 33, "visibility": [0.0] * 33,
                    }
                    heatmap = np.zeros((h, w), dtype=np.uint8)

                # 保存热力图
                heatmap_path = os.path.join(heatmap_dir, f"heatmap_{i+1:05d}.png")
                cv2.imwrite(heatmap_path, heatmap)

                if (i + 1) % 30 == 0:
                    logger.info(f"[MementoPose2D] 进度: {i+1}/{len(frame_files)} 帧")

        # 保存 JSON
        pose_json_path = os.path.join(pose_dir, "keypoints.json")
        with open(pose_json_path, "w") as f:
            json.dump(all_keypoints, f, indent=2)

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "pose_json_path": pose_json_path,
            "heatmap_dir": heatmap_dir,
            "num_pose_frames": len(all_keypoints),
            "keypoint_count": 33,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        detected_frames = sum(
            1 for v in all_keypoints.values() if v["visibility"][0] > 0
        )
        logger.info(
            f"[MementoPose2D] (file-based) 完成: {detected_frames}/{len(all_keypoints)} 帧检测到姿势, "
            f"热力图输出到 {heatmap_dir}"
        )
        return (pose_json_path, heatmap_dir)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def extract(self, frames_dir: str, mask_dir: str,
                min_detection_confidence: float, min_tracking_confidence: float,
                heatmap_sigma: float):
        logger.info(
            f"[MementoPose2D] frames: {frames_dir}, masks: {mask_dir}, "
            f"sigma={heatmap_sigma}, tensor_ops={self._use_tensor_ops}"
        )

        if self._use_tensor_ops and _TENSOR_OPS_AVAILABLE:
            try:
                return self._extract_tensor_ops(
                    frames_dir, mask_dir,
                    min_detection_confidence, min_tracking_confidence,
                    heatmap_sigma,
                )
            except Exception as e:
                logger.warning(
                    f"[MementoPose2D] GPU 张量操作失败: {e}，回退到文件级逻辑"
                )

        return self._extract_file_based(
            frames_dir, mask_dir,
            min_detection_confidence, min_tracking_confidence,
            heatmap_sigma,
        )


NODE_CLASS_MAPPINGS = {"MementoPose2D": MementoPose2D}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPose2D": "Memento 03 - MediaPipe 2D 骨骼 + 热力图 (Mask过滤)"}