"""Memento 02 — SAM3 时序分割节点

输入: 帧序列 + 用户点击坐标 → 输出: 时序一致性分割 Mask
SAM3 = Meta Segment Anything Model 3（统一图像+视频分割，支持概念提示）
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

# SAM3 导入（pip install sam3 预装在容器中）
try:
    from sam3.model_builder import build_sam3_video_model
except ImportError as e:
    logger.error(f"[MementoSegment] SAM3 import failed: {e}")
    raise


class MementoSegment:
    """节点 2: 时序分割 — SAM3 像素级 Mask（时序一致性传播）"""

    # 模型 checkpoint 路径
    CHECKPOINT_PATH = "/models/sam3/sam3.safetensors"

    _predictor = None  # 单例缓存，避免重复加载

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "click_points": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_dir",)
    FUNCTION = "segment"
    CATEGORY = "Memento/02_Segment"

    @classmethod
    def load_model(cls):
        """懒加载 SAM3 视频预测器，缓存单例"""
        if cls._predictor is not None:
            return cls._predictor

        start_time = time.time()

        # 检查模型文件存在
        if not os.path.exists(cls.CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"SAM3 模型不存在: {cls.CHECKPOINT_PATH}\n"
                f"请先运行 bash download_models.sh 下载模型\n"
                f"注意：SAM3 需要先在 HuggingFace facebook/sam3 申请访问权限"
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MementoSegment] 加载 SAM3 到 {device}...")

        # 构建 SAM3 视频模型
        sam3_model = build_sam3_video_model(checkpoint_path=cls.CHECKPOINT_PATH)
        predictor = sam3_model.tracker
        predictor.backbone = sam3_model.detector.backbone
        predictor.to(device)

        elapsed = time.time() - start_time
        logger.info(f"[MementoSegment] SAM3 加载完成，耗时 {elapsed:.1f}s")

        # 显存检查
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoSegment] 显存占用: {mem_used:.2f} GB")
            if mem_used > 6.0:
                logger.warning(
                    f"[MementoSegment] 显存占用超过 6GB 限制: {mem_used:.2f} GB"
                )

        cls._predictor = predictor
        return predictor

    def parse_click_points(self, points_str: str) -> list:
        """解析点击坐标 JSON"""
        try:
            points = json.loads(points_str)
            return points
        except json.JSONDecodeError:
            logger.error(f"[MementoSegment] 点击坐标解析失败: {points_str}")
            raise

    def segment(self, frames_dir: str, click_points: str):
        logger.info(f"[MementoSegment] frames: {frames_dir}")

        # 检查输入目录
        if not os.path.exists(frames_dir):
            raise FileNotFoundError(f"帧目录不存在: {frames_dir}")

        # 获取排序后的帧文件列表
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        # 创建输出目录
        mask_dir = "/workspace/masks"
        Path(mask_dir).mkdir(parents=True, exist_ok=True)

        # 读取第一帧获取尺寸
        first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        if first_frame is None:
            raise RuntimeError(f"无法读取第一帧: {frames_dir}/{frame_files[0]}")
        h, w = first_frame.shape[:2]
        logger.info(f"[MementoSegment] {len(frame_files)} 帧, 尺寸 {w}x{h}")

        # 加载模型
        predictor = self.load_model()

        # 初始化推理状态
        logger.info("[MementoSegment] 初始化推理状态（加载所有帧）...")
        init_start = time.time()
        inference_state = predictor.init_state(video_path=frames_dir)
        logger.info(
            f"[MementoSegment] 推理状态初始化完成，耗时 {time.time() - init_start:.1f}s"
        )

        # 解析点击点（只对第一帧做点击提示）
        points_data = self.parse_click_points(click_points)
        if not points_data:
            raise RuntimeError("没有提供点击坐标，至少需要一个正样本点")

        # SAM3 要求归一化坐标 [0,1]
        point_coords = []
        point_labels = []
        for p in points_data:
            point_coords.append([p["x"] / w, p["y"] / h])
            point_labels.append(p.get("label", 1))

        points_tensor = torch.tensor(point_coords, dtype=torch.float32)
        points_labels_tensor = torch.tensor(point_labels, dtype=torch.int32)

        # 在第一帧添加点击
        logger.info(f"[MementoSegment] 添加 {len(point_coords)} 个点到第一帧")
        _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=points_tensor,
            labels=points_labels_tensor,
            clear_old_points=False,
        )
        logger.info(f"[MementoSegment] 目标 ID: {out_obj_ids}")

        # 时序传播得到所有帧的 mask
        logger.info("[MementoSegment] 开始时序传播...")
        prop_start = time.time()

        saved_count = 0
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=len(frame_files),
            reverse=False,
            propagate_preflight=True,
        ):
            # 取第一个目标的 mask
            if 1 in obj_ids:
                idx_in_list = obj_ids.index(1)
                mask = (video_res_masks[idx_in_list] > 0.0).cpu().numpy()
                mask_uint8 = (mask * 255).astype(np.uint8)
                out_path = os.path.join(mask_dir, f"mask_{frame_idx+1:05d}.png")
                cv2.imwrite(out_path, mask_uint8)
                saved_count += 1

        elapsed = time.time() - prop_start
        logger.info(
            f"[MementoSegment] 时序传播完成，保存 {saved_count} 帧，"
            f"耗时 {elapsed:.1f}s"
        )

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "mask_dir": mask_dir,
            "num_masks": saved_count,
            "mask_height": h,
            "mask_width": w,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(f"[MementoSegment] 全部 masks 输出到 {mask_dir}")
        return (mask_dir,)


NODE_CLASS_MAPPINGS = {"MementoSegment": MementoSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoSegment": "Memento 02 - SAM3 时序分割"}