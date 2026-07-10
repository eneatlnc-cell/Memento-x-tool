"""Memento 03 — MediaPipe 2D 骨骼提取节点

输入: 帧序列 + 掩码目录 → 输出: 33 关键点 JSON（仅 mask 区域）
"""
import logging
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose


class MementoPose2D:
    """节点 3: 2D 骨骼 — MediaPipe 33 关键点提取（仅在 mask 区域检测）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "min_detection_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_tracking_confidence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("pose_json_path",)
    FUNCTION = "extract"
    CATEGORY = "Memento/03_Pose2D"

    def load_mask(self, mask_path: str, target_h: int, target_w: int) -> np.ndarray | None:
        """加载掩码，如果文件不存在返回 None"""
        if not os.path.exists(mask_path):
            return None
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h))
        return mask

    def keypoints_to_dict(self, landmarks, frame_w: int, frame_h: int) -> dict:
        """将 MediaPipe landmarks 转为归一化 JSON 格式"""
        kp = {
            "x": [],
            "y": [],
            "z": [],
            "visibility": [],
        }
        for lm in landmarks.landmark:
            kp["x"].append(round(lm.x, 6))
            kp["y"].append(round(lm.y, 6))
            kp["z"].append(round(lm.z, 6))
            kp["visibility"].append(round(lm.visibility, 6))
        return kp

    def extract(self, frames_dir: str, mask_dir: str, min_detection_confidence: float, min_tracking_confidence: float):
        logger.info(f"[MementoPose2D] frames: {frames_dir}, masks: {mask_dir}")

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
        Path(pose_dir).mkdir(parents=True, exist_ok=True)

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

                # 在 mask 区域内做检测：mask 外区域涂黑
                if mask is not None:
                    masked_frame = frame_rgb.copy()
                    masked_frame[mask == 0] = 0
                    results = pose.process(masked_frame)
                else:
                    results = pose.process(frame_rgb)

                frame_key = f"frame_{i+1:05d}"
                if results.pose_landmarks:
                    valid_kp = self.keypoints_to_dict(results.pose_landmarks, w, h)

                    # 过滤掉不在 mask 区域内的关键点
                    if mask is not None:
                        for j in range(len(valid_kp["x"])):
                            px = int(valid_kp["x"][j] * w)
                            py = int(valid_kp["y"][j] * h)
                            if 0 <= px < w and 0 <= py < h:
                                if mask[py, px] == 0:
                                    valid_kp["visibility"][j] = 0.0

                    all_keypoints[frame_key] = valid_kp
                else:
                    all_keypoints[frame_key] = {
                        "x": [0.0] * 33,
                        "y": [0.0] * 33,
                        "z": [0.0] * 33,
                        "visibility": [0.0] * 33,
                    }

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
            "num_pose_frames": len(all_keypoints),
            "keypoint_count": 33,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        detected_frames = sum(1 for v in all_keypoints.values() if v["visibility"][0] > 0)
        logger.info(
            f"[MementoPose2D] 完成: {detected_frames}/{len(all_keypoints)} 帧检测到姿势, "
            f"输出到 {pose_json_path}"
        )
        return (pose_json_path,)


NODE_CLASS_MAPPINGS = {"MementoPose2D": MementoPose2D}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPose2D": "Memento 03 - MediaPipe 2D 骨骼"}