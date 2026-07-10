"""Memento 02 — SAM2 时序分割节点

输入: 帧序列 + 用户点击坐标 → 输出: 时序一致性分割 Mask
SAM3-Large = SAM2-Hiera-Large（社区命名习惯沿用）
"""
import logging
import json
import os
import time
from pathlib import Path

import torch
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# SAM2 导入（预安装在 /opt/sam2）
try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError as e:
    logger.error(f"[MementoSegment] SAM2 import failed: {e}")
    raise


class MementoSegment:
    """节点 2: 时序分割 — SAM2-Hiera-Large 像素级 Mask（时序一致性传播）"""

    # 模型路径映射
    MODEL_PATHS = {
        "sam3-large": "/models/sam3/sam2_hiera_large.pt",
    }
    MODEL_CFG = {
        "sam3-large": "sam2_hiera_l.yaml",
    }

    _predictor = None  # 单例缓存，避免重复加载

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "click_points": ("STRING", {"default": "[]", "multiline": False}),
                "model": (["sam3-large",], {"default": "sam3-large"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("mask_dir",)
    FUNCTION = "segment"
    CATEGORY = "Memento/02_Segment"

    @classmethod
    def load_model(cls, model_name: str):
        """懒加载模型，缓存单例"""
        if cls._predictor is not None:
            return cls._predictor
        
        start_time = time.time()
        checkpoint_path = cls.MODEL_PATHS[model_name]
        cfg_name = cls.MODEL_CFG[model_name]
        
        # 检查模型文件存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"SAM2 模型不存在: {checkpoint_path}\n"
                f"请先运行 bash download_models.sh 下载模型"
            )
        
        # 模型配置文件路径（SAM2 安装时已经拷贝）
        if os.path.exists(f"/models/sam3/{cfg_name}"):
            cfg_path = f"/models/sam3/{cfg_name}"
        else:
            cfg_path = f"/opt/sam2/sam2/configs/{cfg_name}"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[MementoSegment] 加载 {model_name} 到 {device}...")
        
        predictor = build_sam2_video_predictor(
            cfg_path, checkpoint_path, device=device
        )
        
        elapsed = time.time() - start_time
        logger.info(f"[MementoSegment] 模型加载完成，耗时 {elapsed:.1f}s")
        
        # 显存检查
        if torch.cuda.is_available():
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoSegment] 显存占用: {mem_used:.2f} GB")
            if mem_used > 6.0:
                logger.warning(f"[MementoSegment] 显存占用超过 6GB 限制: {mem_used:.2f} GB")
        
        cls._predictor = predictor
        return predictor

    def parse_click_points(self, points_str: str) -> list:
        """解析点击坐标 JSON"""
        try:
            points = json.loads(points_str)
            # 格式: [{"x": 123, "y": 456, "label": 1}, ...]
            # label 1 = 正样本（前景）, 0 = 负样本（背景）
            return points
        except json.JSONDecodeError:
            logger.error(f"[MementoSegment] 点击坐标解析失败: {points_str}")
            raise

    def segment(self, frames_dir: str, click_points: str, model: str):
        logger.info(f"[MementoSegment] frames: {frames_dir}, model: {model}")
        
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
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        first_frame = cv2.imread(first_frame_path)
        h, w = first_frame.shape[:2]
        logger.info(f"[MementoSegment] {len(frame_files)} 帧, 尺寸 {w}x{h}")
        
        # 加载模型
        predictor = self.load_model(model)
        
        # 初始化推理状态
        inference_state = predictor.init_state(video_path=frames_dir)
        logger.info("[MementoSegment] 推理状态初始化完成")
        
        # 解析点击点（只对第一帧做点击提示）
        points_data = self.parse_click_points(click_points)
        if not points_data:
            raise RuntimeError("没有提供点击坐标，至少需要一个正样本点")
        
        # 转换为 SAM2 格式
        point_coords = []
        point_labels = []
        for p in points_data:
            point_coords.append([p["x"], p["y"]])
            point_labels.append(p.get("label", 1))
        
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)
        
        # 在第一帧添加点击
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points=point_coords,
            labels=point_labels,
        )
        logger.info(f"[MementoSegment] 添加 {len(point_coords)} 个点到第一帧")
        
        # 时序传播得到所有帧的 mask
        logger.info("[MementoSegment] 开始时序传播...")
        start_time = time.time()
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # 取第一个目标的 mask
            mask = (out_mask_logits[out_obj_ids.index(1)] > 0).cpu().numpy()
            # 保存为 PNG（二值）
            mask_uint8 = (mask * 255).astype(np.uint8)
            out_path = os.path.join(mask_dir, f"mask_{out_frame_idx+1:05d}.png")
            cv2.imwrite(out_path, mask_uint8)
        
        elapsed = time.time() - start_time
        logger.info(f"[MementoSegment] 时序传播完成，{len(frame_files)} 帧耗时 {elapsed:.1f}s")
        
        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)
        
        context.update({
            "mask_dir": mask_dir,
            "num_masks": len(frame_files),
            "mask_height": h,
            "mask_width": w,
        })
        
        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)
        
        logger.info(f"[MementoSegment] 全部 masks 输出到 {mask_dir}")
        return (mask_dir,)


NODE_CLASS_MAPPINGS = {"MementoSegment": MementoSegment}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoSegment": "Memento 02 - SAM2 时序分割 (SAM3-Large)"}