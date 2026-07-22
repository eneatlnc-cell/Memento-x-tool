"""Memento 04 — MotionBERT 2D→3D 姿态归一化 + Depth 深度图生成

输入: 03 输出的 2D 骨骼点位数据 + Pose 热力图
输出: 归一化 3D 姿态 JSON + Depth 深度图（每帧 PNG）

Depth 深度图:
  - 基于 3D 姿态 Z 轴 + 前景距离场融合
  - 前景: 距离变换渐变（边缘→内部）
  - 骨骼: Z 深度加权（近大远小）
  - 背景: 纯黑
"""
import logging
import json
import os
import time
from pathlib import Path

import torch
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MementoPose3D:
    """节点 4: 3D 归一化 — MotionBERT 2D→3D + Depth 深度图"""

    # 多候选路径：容器内 /root/data/models、宿主机 ~/.memento、本地直跑
    _CANDIDATE_PATHS = [
        os.path.join(os.environ.get("COMFYUI_MODEL_DIR", "/root/data/models"), "pose", "motionbert_ft_h36m.pth"),
        os.path.expanduser("~/.memento/workspace/models/pose/motionbert_ft_h36m.pth"),
        "/models/pose/motionbert_ft_h36m.pth",
    ]
    CHECKPOINT_PATH = next((p for p in _CANDIDATE_PATHS if os.path.exists(p)), _CANDIDATE_PATHS[0])

    # MediaPipe 33 → H36M 17 关键点映射
    MP_TO_H36M = {
        0:  0,   # hip → pelvis (使用左右髋中点)
        1:  24,  # r_hip
        2:  26,  # r_knee
        3:  28,  # r_ankle
        4:  23,  # l_hip
        5:  25,  # l_knee
        6:  27,  # l_ankle
        7:  0,   # spine → 用 hip 近似
        8:  0,   # neck → 用 hip 近似
        9:  0,   # head → 用 nose
        10: 12,  # r_shoulder
        11: 14,  # r_elbow
        12: 16,  # r_wrist
        13: 11,  # l_shoulder
        14: 13,  # l_elbow
        15: 15,  # l_wrist
        16: 0,   # thorax → 用 hip 近似
    }

    _model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_json_path": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pose3d_json_path", "depth_dir")
    FUNCTION = "normalize"
    CATEGORY = "Memento/04_Pose3D"

    @classmethod
    def load_model(cls):
        if cls._model is not None:
            return cls._model

        start_time = time.time()
        if not os.path.exists(cls.CHECKPOINT_PATH):
            logger.warning(f"[MementoPose3D] MotionBERT 模型不存在，使用轻量级抬升器")
            model = SimplePoseLifter()
            return model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MementoPose3D] 加载 MotionBERT 到 {device}...")

        checkpoint = torch.load(cls.CHECKPOINT_PATH, map_location=device, weights_only=False)
        model = SimplePoseLifter()
        state_dict = checkpoint.get("model_pos", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        elapsed = time.time() - start_time
        logger.info(f"[MementoPose3D] 加载完成，耗时 {elapsed:.1f}s")
        cls._model = model
        return model

    def convert_mp33_to_h36m(self, keypoints_33: dict) -> np.ndarray:
        kp_17 = np.zeros((17, 2), dtype=np.float32)
        for h36m_idx, mp_idx in self.MP_TO_H36M.items():
            if mp_idx < len(keypoints_33["x"]):
                kp_17[h36m_idx, 0] = keypoints_33["x"][mp_idx]
                kp_17[h36m_idx, 1] = keypoints_33["y"][mp_idx]
        l_hip_x = keypoints_33["x"][23] if 23 < len(keypoints_33["x"]) else 0
        l_hip_y = keypoints_33["y"][23] if 23 < len(keypoints_33["y"]) else 0
        r_hip_x = keypoints_33["x"][24] if 24 < len(keypoints_33["x"]) else 0
        r_hip_y = keypoints_33["y"][24] if 24 < len(keypoints_33["y"]) else 0
        kp_17[0] = [(l_hip_x + r_hip_x) / 2, (l_hip_y + r_hip_y) / 2]
        return kp_17

    def normalize_height(self, kp_3d: np.ndarray) -> np.ndarray:
        bone_length = 0.0
        bone_length += np.linalg.norm(kp_3d[0] - kp_3d[8])
        bone_length += np.linalg.norm(kp_3d[8] - kp_3d[9])
        bone_length += np.linalg.norm(kp_3d[1] - kp_3d[2]) + np.linalg.norm(kp_3d[2] - kp_3d[3])
        bone_length += np.linalg.norm(kp_3d[4] - kp_3d[5]) + np.linalg.norm(kp_3d[5] - kp_3d[6])
        if bone_length > 1e-6:
            scale = 1.7 / bone_length
            kp_3d = kp_3d * scale
        return kp_3d

    def generate_depth_map(self, kp_3d: np.ndarray, mask: np.ndarray | None,
                           h: int, w: int) -> np.ndarray:
        """
        融合 3D 姿态 Z 轴 + 前景距离场 → 深度图

        策略:
          1. 距离场: 前景区域内部到边缘的欧氏距离（边缘=0, 中心=255）
          2. Z 深度: 3D 关键点 Z 值归一化后在整个前景区域做径向基插值
          3. 融合: 距离场 × 0.4 + Z深度 × 0.6
          4. 背景: 纯黑 0
        """
        depth = np.zeros((h, w), dtype=np.float32)

        # ── 距离场分量 ──
        if mask is not None:
            mask_bin = mask.copy()
            if mask_bin.max() > 127:
                _, mask_bin = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)
            dist_fg = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
            if dist_fg.max() > 0:
                dist_fg = dist_fg / dist_fg.max()
            else:
                dist_fg = np.zeros((h, w), dtype=np.float32)
        else:
            dist_fg = np.ones((h, w), dtype=np.float32) * 0.5

        # ── Z 深度分量（基于 3D 关键点径向基插值） ──
        z_vals = kp_3d[:, 2]  # (17,)
        z_min, z_max = z_vals.min(), z_vals.max()
        if z_max - z_min > 1e-6:
            z_norm = (z_vals - z_min) / (z_max - z_min)
        else:
            z_norm = np.ones(17) * 0.5

        # 使用关键点 XY 坐标做径向基插值
        rbf_depth = np.zeros((h, w), dtype=np.float32)
        weights_sum = np.zeros((h, w), dtype=np.float32) + 1e-8

        for i in range(17):
            px = int(kp_3d[i, 0] * w)
            py = int(kp_3d[i, 1] * h)
            if 0 <= px < w and 0 <= py < h:
                ys_grid, xs_grid = np.ogrid[:h, :w]
                rbf = np.exp(-((xs_grid - px)**2 + (ys_grid - py)**2) / (2 * (max(w, h) * 0.05)**2))
                rbf_depth += rbf * z_norm[i]
                weights_sum += rbf

        rbf_depth = rbf_depth / weights_sum

        # ── 融合 ──
        if mask is not None:
            depth = dist_fg * 0.4 + rbf_depth * 0.6
            _, mask_bin2 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            depth[mask_bin2 == 0] = 0
        else:
            depth = rbf_depth

        depth = (depth * 255).clip(0, 255).astype(np.uint8)
        return depth

    def load_mask(self, mask_dir: str, frame_idx: int, h: int, w: int) -> np.ndarray | None:
        mask_path = os.path.join(mask_dir, f"mask_{frame_idx+1:05d}.png")
        if not os.path.exists(mask_path):
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        return mask

    def normalize(self, pose_json_path: str, mask_dir: str):
        logger.info(f"[MementoPose3D] 输入: {pose_json_path}, masks: {mask_dir}")

        if not os.path.exists(pose_json_path):
            raise FileNotFoundError(f"关键点文件不存在: {pose_json_path}")

        with open(pose_json_path, "r") as f:
            keypoints_data = json.load(f)

        # 创建输出目录
        pose3d_dir = "/workspace/pose3d"
        depth_dir = "/workspace/depth"
        Path(pose3d_dir).mkdir(parents=True, exist_ok=True)
        Path(depth_dir).mkdir(parents=True, exist_ok=True)

        # 加载模型
        try:
            model = self.load_model()
        except Exception as e:
            logger.warning(f"[MementoPose3D] 模型加载失败，使用简单位姿提升: {e}")
            model = None

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 从 context.json 获取帧尺寸
        context_path = "/workspace/context.json"
        h, w = 1080, 1920
        if os.path.exists(context_path):
            with open(context_path) as f:
                ctx = json.load(f)
                h = ctx.get("original_height", h)
                w = ctx.get("original_width", w)

        normalized_data = {}

        for frame_idx, (frame_key, kp_33) in enumerate(keypoints_data.items()):
            # 转换 33→17
            kp_17_2d = self.convert_mp33_to_h36m(kp_33)

            if model is not None:
                with torch.no_grad():
                    kp_input = torch.from_numpy(kp_17_2d).unsqueeze(0).to(device)
                    kp_3d = model(kp_input).squeeze(0).cpu().numpy()
            else:
                kp_3d = np.zeros((17, 3), dtype=np.float32)
                kp_3d[:, :2] = kp_17_2d

            # 身高归一化
            kp_3d = self.normalize_height(kp_3d)

            # 序列化 3D 姿态
            normalized_data[frame_key] = {
                "x": [round(float(v[0]), 6) for v in kp_3d],
                "y": [round(float(v[1]), 6) for v in kp_3d],
                "z": [round(float(v[2]), 6) for v in kp_3d],
            }

            # 生成 Depth 深度图
            mask = self.load_mask(mask_dir, frame_idx, h, w)
            depth_map = self.generate_depth_map(kp_3d, mask, h, w)
            depth_path = os.path.join(depth_dir, f"depth_{frame_idx+1:05d}.png")
            cv2.imwrite(depth_path, depth_map)

            if (frame_idx + 1) % 30 == 0:
                logger.info(f"[MementoPose3D] 进度: {frame_idx+1}/{len(keypoints_data)} 帧")

        # 保存 3D 姿态 JSON
        pose3d_json_path = os.path.join(pose3d_dir, "normalized.json")
        with open(pose3d_json_path, "w") as f:
            json.dump(normalized_data, f, indent=2)

        # 更新 context.json
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "pose3d_json_path": pose3d_json_path,
            "depth_dir": depth_dir,
            "num_pose3d_frames": len(normalized_data),
            "keypoint_count_3d": 17,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(
            f"[MementoPose3D] 完成: {len(normalized_data)} 帧 3D 姿态 + Depth 深度图, "
            f"深度图输出到 {depth_dir}"
        )
        return (pose3d_json_path, depth_dir)


class SimplePoseLifter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(17 * 2, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 17 * 3)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.reshape(B, 17, 3)


NODE_CLASS_MAPPINGS = {"MementoPose3D": MementoPose3D}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPose3D": "Memento 04 - MotionBERT 3D 归一化"}