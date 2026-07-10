"""Memento 控制信号提取工具 — control_extractor.py

从 QuadMask + 帧数据 + 3D 姿态中提取 LTX-Video IC-LoRA 所需的控制信号：
- Pose Video: 骨架绘制视频（白色骨架，黑色背景）
- Depth Video: 伪深度图（前景白 → 背景黑渐变）
- Canny Video: 边缘检测视频（仅在 mask 区域内保留）

IC-LoRA 控制信号格式：MP4 视频，尺寸与原始帧一致
"""
import logging
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── 骨架连接定义（17 关键点 H36M 格式） ──
SKELETON_CONNECTIONS = [
    (0, 1), (0, 4),          # 髋 → 左右髋
    (1, 2), (2, 3),           # 右腿
    (4, 5), (5, 6),           # 左腿
    (0, 7), (7, 8), (8, 9),   # 脊柱 → 颈 → 头
    (8, 10), (10, 11), (11, 12),  # 右臂
    (8, 13), (13, 14), (14, 15),  # 左臂
    (7, 16),                   # 脊柱 → 胸
]

SKELETON_COLOR = (255, 255, 255)    # 白色骨架
KEYPOINT_COLOR = (255, 255, 255)    # 白色关键点
KEYPOINT_RADIUS = 3


def draw_skeleton(frame: np.ndarray, kp_3d: dict, h: int, w: int) -> np.ndarray:
    """在黑色背景上绘制骨架图"""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    xs = kp_3d["x"]
    ys = kp_3d["y"]

    # 绘制骨架连线
    for conn in SKELETON_CONNECTIONS:
        i, j = conn
        if i < len(xs) and j < len(xs):
            x1 = int(xs[i] * w)
            y1 = int(ys[i] * w)
            x2 = int(xs[j] * w)
            y2 = int(ys[j] * w)
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(canvas, (x1, y1), (x2, y2), SKELETON_COLOR, 2)

    # 绘制关键点
    for i in range(len(xs)):
        x = int(xs[i] * w)
        y = int(ys[i] * w)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

    return canvas


def generate_pose_video(
    pose3d_json_path: str,
    output_mp4: str,
    h: int,
    w: int,
    num_frames: int,
    fps: int = 25,
) -> str:
    """从 3D 姿态 JSON 生成骨架视频 (MP4)

    Args:
        pose3d_json_path: 3D 姿态 JSON 路径
        output_mp4: 输出 MP4 路径
        h, w: 帧尺寸
        num_frames: 帧数
        fps: 帧率

    Returns:
        输出 MP4 路径
    """
    with open(pose3d_json_path, "r") as f:
        pose_data = json.load(f)

    # 创建临时帧目录
    tmp_dir = str(Path(output_mp4).parent / ".pose_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(num_frames):
        frame_key = f"frame_{i+1:05d}"
        kp = pose_data.get(frame_key, {"x": [0]*17, "y": [0]*17, "z": [0]*17})
        skeleton = draw_skeleton(kp, h, w)
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), skeleton)

    # 用 FFmpeg 合成 MP4
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_mp4
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 合成 pose 视频失败: {result.stderr}")

    # 清理临时文件
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"[ControlExtractor] Pose 视频生成: {output_mp4}")
    return output_mp4


def generate_depth_video(
    mask_dir: str,
    output_mp4: str,
    h: int,
    w: int,
    num_frames: int,
    fps: int = 25,
) -> str:
    """从掩码生成伪深度视频 (MP4)

    深度图：前景区域 → 白色渐变（距离边缘越远越亮），背景 → 黑色

    Args:
        mask_dir: 掩码目录
        output_mp4: 输出 MP4 路径
        h, w: 帧尺寸
        num_frames: 帧数
        fps: 帧率

    Returns:
        输出 MP4 路径
    """
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith('.png')
    ])

    tmp_dir = str(Path(output_mp4).parent / ".depth_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(min(num_frames, len(mask_files))):
        mask_path = os.path.join(mask_dir, mask_files[i])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            depth = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (w, h))
            _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            # 距离变换：前景内部距离
            depth = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
            if depth.max() > 0:
                depth = (depth / depth.max() * 255).astype(np.uint8)
            else:
                depth = np.zeros((h, w), dtype=np.uint8)

        # 转为 3 通道
        depth_rgb = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), depth_rgb)

    # 补充剩余帧
    for i in range(len(mask_files), num_frames):
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), blank)

    # FFmpeg 合成
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_mp4
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 合成 depth 视频失败: {result.stderr}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"[ControlExtractor] Depth 视频生成: {output_mp4}")
    return output_mp4


def generate_canny_video(
    frames_dir: str,
    mask_dir: str,
    output_mp4: str,
    h: int,
    w: int,
    num_frames: int,
    fps: int = 25,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> str:
    """从帧 + 掩码生成 Canny 边缘视频 (MP4)

    仅在 mask 区域内保留边缘

    Args:
        frames_dir: 帧目录
        mask_dir: 掩码目录
        output_mp4: 输出 MP4 路径
        h, w: 帧尺寸
        num_frames: 帧数
        fps: 帧率
        low_threshold: Canny 低阈值
        high_threshold: Canny 高阈值

    Returns:
        输出 MP4 路径
    """
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith('.png')
    ])

    tmp_dir = str(Path(output_mp4).parent / ".canny_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    for i in range(min(num_frames, len(frame_files))):
        frame_path = os.path.join(frames_dir, frame_files[i])
        frame = cv2.imread(frame_path)
        if frame is None:
            canny = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, (w, h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Canny 边缘检测
            edges = cv2.Canny(gray, low_threshold, high_threshold)

            # 加载掩码，仅在 mask 内保留边缘
            if i < len(mask_files):
                mask_path = os.path.join(mask_dir, mask_files[i])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (w, h))
                    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    edges = cv2.bitwise_and(edges, mask_bin)

            canny = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), canny)

    # 补充剩余帧
    for i in range(len(frame_files), num_frames):
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), blank)

    # FFmpeg 合成
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(tmp_dir, "frame_%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_mp4
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 合成 canny 视频失败: {result.stderr}")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"[ControlExtractor] Canny 视频生成: {output_mp4}")
    return output_mp4


def align_resolution(h: int, w: int) -> tuple[int, int]:
    """将分辨率对齐到 64 的倍数（LTX-Video 要求）"""
    h_aligned = ((h + 63) // 64) * 64
    w_aligned = ((w + 63) // 64) * 64
    return h_aligned, w_aligned


def align_frames(num_frames: int, target_format: str = "8n+1") -> int:
    """将帧数对齐到 LTX-Video 要求（8n+1）"""
    remainder = (num_frames - 1) % 8
    if remainder != 0:
        num_frames = num_frames + (8 - remainder)
    return num_frames