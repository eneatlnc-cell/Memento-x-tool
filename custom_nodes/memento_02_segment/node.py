"""Memento 02 — SAM2.1 视频时序分割节点

输入: 01 输出的 30fps 原始帧
输出: 逐帧人物蒙版 Mask（四层 SVG 遮罩底层数据源）

四层 SVG 遮罩结构:
  Layer 0: 人物前景蒙版（二值，255/0）
  Layer 1: 边缘羽化层（高斯模糊 3px）
  Layer 2: 发丝/细节层（高阈值边缘保留）
  Layer 3: 半透明区域（烟雾/玻璃/动态模糊）

SAM2.1 = Meta Segment Anything Model 2.1（Apache 2.0，完全开放，无需申请）
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

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    logger.error(f"[MementoSegment] SAM2 import failed: {e}")
    raise


class MementoSegment:
    """节点 2: 时序分割 — SAM2.1 像素级 Mask + 四层 SVG 遮罩"""

    CHECKPOINT_PATH = os.path.join(os.environ.get("COMFYUI_MODEL_DIR", "/root/data/models"), "sam2", "sam2.1_hiera_large.pt")
    MODEL_CFG = "sam2.1_hiera_l.yaml"
    _predictor = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "click_points": ("STRING", {"default": "[]", "multiline": False}),
                "score_threshold_detection": ("FLOAT", {
                    "default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "SAM2 检测置信度阈值，降低可检测更多目标",
                }),
                "pred_iou_thresh": ("FLOAT", {
                    "default": 0.88, "min": 0.5, "max": 1.0, "step": 0.02,
                    "tooltip": "掩码质量阈值，降低保留更多掩码",
                }),
                "stability_score_thresh": ("FLOAT", {
                    "default": 0.95, "min": 0.5, "max": 1.0, "step": 0.01,
                    "tooltip": "掩码稳定性阈值，提高得到更稳定边缘",
                }),
                "recondition_every_nth_frame": ("INT", {
                    "default": 16, "min": 0, "max": 60, "step": 1,
                    "tooltip": "重条件化间隔帧数，0=禁用，值越小追踪越稳定",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("mask_dir", "svg_mask_dir")
    FUNCTION = "segment"
    CATEGORY = "Memento/02_Segment"

    @classmethod
    def load_model(cls):
        if cls._predictor is not None:
            return cls._predictor

        start_time = time.time()
        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"SAM2.1 模型不存在: {cls.CHECKPOINT_PATH}\n"
                f"请先运行 bash download_models.sh 下载模型"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MementoSegment] 加载 SAM2.1 到 {device}...")

        predictor = build_sam2_video_predictor(
            cls.MODEL_CFG, cls.CHECKPOINT_PATH, device=device,
        )

        elapsed = time.time() - start_time
        logger.info(f"[MementoSegment] SAM2.1 加载完成，耗时 {elapsed:.1f}s")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoSegment] 显存占用: {mem_used:.2f} GB")

        cls._predictor = predictor
        return predictor

    def parse_click_points(self, points_str: str) -> list:
        try:
            return json.loads(points_str)
        except json.JSONDecodeError:
            logger.error(f"[MementoSegment] 点击坐标解析失败: {points_str}")
            raise

    def generate_four_layer_svg_mask(self, mask_bin: np.ndarray) -> dict:
        h, w = mask_bin.shape
        layers = {}

        layers["layer_0_foreground"] = mask_bin.copy()

        mask_float = mask_bin.astype(np.float32) / 255.0
        feather = cv2.GaussianBlur(mask_float, (7, 7), 3.0)
        layers["layer_1_feather"] = (feather * 255).astype(np.uint8)

        laplacian = cv2.Laplacian(mask_float, cv2.CV_32F, ksize=3)
        laplacian = np.abs(laplacian)
        detail = np.clip(laplacian * 5.0, 0, 1.0)
        layers["layer_2_detail"] = (detail * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(mask_bin, kernel, iterations=1)
        eroded = cv2.erode(mask_bin, kernel, iterations=1)
        semitrans = cv2.subtract(dilated, eroded)
        layers["layer_3_semitrans"] = semitrans

        return layers

    def segment(self, frames_dir: str, click_points: str):
        logger.info(f"[MementoSegment] frames: {frames_dir}")

        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"帧目录不存在: {frames_dir}")

        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        mask_dir = "/workspace/masks"
        svg_mask_dir = "/workspace/masks_svg"
        Path(mask_dir).mkdir(parents=True, exist_ok=True)
        Path(svg_mask_dir).mkdir(parents=True, exist_ok=True)

        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            raise RuntimeError(f"无法读取第一帧: {frames_dir}/{frame_files[0]}")
        h, w = first_frame.shape[:2]
        logger.info(f"[MementoSegment] {len(frame_files)} 帧, 尺寸 {w}x{h}")

        predictor = self.load_model()

        logger.info("[MementoSegment] 初始化推理状态（加载所有帧）...")
        init_start = time.time()
        inference_state = predictor.init_state(video_path=frames_dir)
        logger.info(
            f"[MementoSegment] 推理状态初始化完成，耗时 {time.time() - init_start:.1f}s"
        )

        points_data = self.parse_click_points(click_points)
        if not points_data:
            raise RuntimeError("没有提供点击坐标，至少需要一个正样本点")

        point_coords = []
        point_labels = []
        for p in points_data:
            point_coords.append([p["x"], p["y"]])
            point_labels.append(p.get("label", 1))

        logger.info(f"[MementoSegment] 添加 {len(point_coords)} 个点到第一帧")
        _, out_obj_ids, masks = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )

        logger.info("[MementoSegment] 开始时序传播...")
        prop_start = time.time()
        saved_count = 0

        for frame_idx, obj_ids, out_masks in predictor.propagate_in_video(inference_state):
            if 1 in obj_ids:
                idx_in_list = obj_ids.index(1)
                mask = (out_masks[idx_in_list] > 0.0).cpu().numpy()
                if mask.ndim == 3:
                    mask = mask[0]
                mask_uint8 = (mask * 255).astype(np.uint8)

                out_path = os.path.join(mask_dir, f"mask_{frame_idx+1:05d}.png")
                cv2.imwrite(out_path, mask_uint8)

                svg_layers = self.generate_four_layer_svg_mask(mask_uint8)
                svg_out = os.path.join(svg_mask_dir, f"svg_{frame_idx+1:05d}.npz")
                np.savez_compressed(svg_out, **svg_layers)

                saved_count += 1

        predictor.reset_state(inference_state)

        elapsed = time.time() - prop_start
        logger.info(
            f"[MementoSegment] 时序传播完成，保存 {saved_count} 帧 "
            f"(含四层 SVG 遮罩)，耗时 {elapsed:.1f}s"
        )

        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "mask_dir": mask_dir,
            "svg_mask_dir": svg_mask_dir,
            "num_masks": saved_count,
            "mask_height": h,
            "mask_width": w,
            "svg_layers": ["foreground", "feather", "detail", "semitrans"],
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(f"[MementoSegment] 全部 masks 输出到 {mask_dir}")
        return (mask_dir, svg_mask_dir)


NODE_CLASS_MAPPINGS = {"MementoSegment": MementoSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoSegment": "Memento 02 - SAM2.1 时序分割"}
