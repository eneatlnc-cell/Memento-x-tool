"""Memento Pipeline Runner — GPU 张量全流程编排器

核心架构:
  01 FFmpeg 流式解码 → 30帧分片
  ├─ 分片循环 (02→03→04→05→06)
  │   ├─ 02 SAM3 分割 (GPU tensor in/out)
  │   ├─ 03 MediaPipe 2D 姿态 (GPU tensor in/out)
  │   ├─ 04 MotionBERT 3D (GPU tensor in/out)
  │   ├─ 05 控制信号对齐 (GPU tensor in/out)
  │   └─ 06 LTX 局部重绘 (GPU tensor in/out)
  │   └─ 释放 Mask/Pose/Depth 张量 + torch.cuda.empty_cache()
  ├─ 合并所有 LTX 张量 → 完整序列
  └─ 全局后处理 (07→08→09)
      ├─ 07 RAFT 光流矫正
      ├─ 08 分层光影融合
      └─ 09 FFmpeg 成片

收益:
  - 无中间图片落地 → 消除磁盘 I/O 延迟 (~40-70% 速度提升)
  - 无编解码二次像素误差 → 截断误差累积
  - 显存内张量流转 → 内存高效
"""
import logging
import time
import json
import os
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from .stream_decoder import StreamDecoder
from .ops import (
    segment_video,
    extract_pose_2d,
    lift_pose_3d,
    align_controls,
    get_svg_mask_data,
    clear_model_cache,
)
from .ops.sub import (
    ltx_inpaint,
    raft_correct,
    fusion_blend,
    composite_video,
)

logger = logging.getLogger(__name__)

# ── 配置 ──
CHUNK_SIZE = 30                # 每分片帧数 (1秒@30fps)
GPU_MEMORY_CLEANUP = True      # 每分片完成后释放显存
OUTPUT_DIR = "/workspace"


class PipelineRunner:
    """Memento 全流程编排器 — GPU 张量流 + 分片推理"""

    def __init__(
        self,
        video_path: str,
        click_points: list = None,
        reference_dir: str = "",
        prompt: str = "",
        control_mode: str = "pose+depth+canny",
        control_strength: float = 0.7,
        ingredients_enabled: bool = True,
        ingredients_strength: float = 1.4,
        num_inference_steps: int = 30,
        seed: int = 42,
        output_dir: str = OUTPUT_DIR,
    ):
        self.video_path = video_path
        self.click_points = click_points or []
        self.reference_dir = reference_dir
        self.prompt = prompt
        self.control_mode = control_mode
        self.control_strength = control_strength
        self.ingredients_enabled = ingredients_enabled
        self.ingredients_strength = ingredients_strength
        self.num_inference_steps = num_inference_steps
        self.seed = seed
        self.output_dir = output_dir

        self.decoder: Optional[StreamDecoder] = None
        self.metadata: dict = {}
        self.audio_path: str = ""

        # 收集所有 LTX 输出帧
        self.all_ltx_frames: list[torch.Tensor] = []
        self.all_original_frames: list[torch.Tensor] = []
        self.all_masks: list[torch.Tensor] = []

        # 统计
        self.stats = {
            "total_chunks": 0,
            "total_frames": 0,
            "chunk_times": [],
            "stage_times": {},
        }

    # ════════════════════════════════════════════════════════════
    # 主入口
    # ════════════════════════════════════════════════════════════

    def run(self) -> str:
        """
        执行完整管线。

        Returns:
            str: 输出视频路径
        """
        total_start = time.time()
        logger.info("=" * 60)
        logger.info("Memento Pipeline Runner v3.0 — GPU 张量流")
        logger.info(f"  视频: {self.video_path}")
        logger.info(f"  分片大小: {CHUNK_SIZE} 帧")
        logger.info(f"  控制模式: {self.control_mode}")
        logger.info("=" * 60)

        # ── 阶段 1: 预处理 ──
        self._stage_01_preprocess()

        # ── 阶段 2: 分片循环 02→06 ──
        self._stage_chunk_loop()

        # ── 阶段 3: 合并 + 全局后处理 07→09 ──
        output_path = self._stage_post_process()

        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info(f"管线完成! 总耗时: {total_elapsed:.1f}s")
        logger.info(f"  分片数: {self.stats['total_chunks']}")
        logger.info(f"  总帧数: {self.stats['total_frames']}")
        if self.stats["chunk_times"]:
            avg_chunk = sum(self.stats["chunk_times"]) / len(self.stats["chunk_times"])
            logger.info(f"  平均分片耗时: {avg_chunk:.1f}s")
        logger.info(f"  输出: {output_path}")
        logger.info("=" * 60)

        return output_path

    # ════════════════════════════════════════════════════════════
    # 阶段 1: 预处理
    # ════════════════════════════════════════════════════════════

    def _stage_01_preprocess(self):
        """01: FFmpeg 流式解码准备 + 音频提取 + 元数据"""
        t0 = time.time()
        logger.info("[阶段 1/3] 预处理: 流式解码 + 音频提取 + 元数据")

        self.decoder = StreamDecoder(self.video_path)
        self.metadata = self.decoder.get_metadata()
        self.audio_path = self.decoder.extract_audio(self.output_dir)

        # 保存元数据
        meta_path = os.path.join(self.output_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(
            f"  视频: {self.metadata['width']}x{self.metadata['height']}, "
            f"{self.metadata['fps']}fps, {self.metadata['nb_frames']}帧"
        )
        if self.audio_path:
            logger.info(f"  音频: {self.audio_path}")
        else:
            logger.info("  音频: 无")

        self.stats["stage_times"]["01_preprocess"] = time.time() - t0

    # ════════════════════════════════════════════════════════════
    # 阶段 2: 分片循环
    # ════════════════════════════════════════════════════════════

    def _stage_chunk_loop(self):
        """02→06 分片循环: 每 30 帧一组，GPU 张量流转"""
        t0 = time.time()
        logger.info("[阶段 2/3] 分片循环: 02→03→04→05→06 (GPU 张量流)")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        total_frames = self.metadata.get("nb_frames", 0)

        for chunk_idx, frames_chunk in self.decoder.decode_chunks(CHUNK_SIZE):
            chunk_start = time.time()
            n_frames = frames_chunk.shape[0]
            self.stats["total_frames"] += n_frames

            logger.info(
                f"  ── 分片 {chunk_idx + 1}: {n_frames} 帧 "
                f"({frames_chunk.shape[3]}x{frames_chunk.shape[2]}) ──"
            )

            frames_chunk = frames_chunk.to(device)

            # ── 02 SAM3 分割 ──
            t = time.time()
            if self.click_points:
                masks = segment_video(frames_chunk, self.click_points, device)
            else:
                # 无点击点 → 全图蒙版
                _, _, h, w = frames_chunk.shape
                masks = torch.ones(n_frames, 1, h, w, device=device)
            self.stats["stage_times"].setdefault("02_segment", 0)
            self.stats["stage_times"]["02_segment"] += time.time() - t
            logger.info(f"    02 SAM3: {masks.shape} ({(time.time()-t):.1f}s)")

            # ── 03 MediaPipe 2D 姿态 ──
            t = time.time()
            keypoints_dict, heatmaps = extract_pose_2d(frames_chunk, masks)
            self.stats["stage_times"].setdefault("03_pose2d", 0)
            self.stats["stage_times"]["03_pose2d"] += time.time() - t
            det_frames = sum(
                1 for v in keypoints_dict.values() if v["visibility"][0] > 0
            )
            logger.info(f"    03 Pose2D: {heatmaps.shape} ({det_frames}/{n_frames} 检测到)")

            # ── 04 MotionBERT 3D ──
            t = time.time()
            pose3d_dict, depth_maps = lift_pose_3d(keypoints_dict, masks)
            self.stats["stage_times"].setdefault("04_pose3d", 0)
            self.stats["stage_times"]["04_pose3d"] += time.time() - t
            logger.info(f"    04 Pose3D: {depth_maps.shape}")

            # ── 05 控制信号对齐 ──
            t = time.time()
            control_pack = align_controls(frames_chunk, masks, heatmaps, depth_maps)
            self.stats["stage_times"].setdefault("05_align", 0)
            self.stats["stage_times"]["05_align"] += time.time() - t
            logger.info(f"    05 Align: {control_pack.shape} (Canny/Distance/Pose/Temporal)")

            # ── 06 LTX 局部重绘 ──
            t = time.time()
            ltx_frames = ltx_inpaint(
                frames_chunk, masks, control_pack,
                self.reference_dir, self.prompt, self.metadata,
                self.control_strength, self.ingredients_enabled,
                self.ingredients_strength, self.num_inference_steps, self.seed,
            )
            self.stats["stage_times"].setdefault("06_ltx", 0)
            self.stats["stage_times"]["06_ltx"] += time.time() - t
            logger.info(f"    06 LTX: {ltx_frames.shape}")

            # ── 持久保存 LTX 帧 (仅此处落地) ──
            self.all_ltx_frames.append(ltx_frames.cpu())
            self.all_original_frames.append(frames_chunk.cpu())
            self.all_masks.append(masks.cpu())

            # ── 释放 GPU 显存 ──
            if GPU_MEMORY_CLEANUP:
                del masks, heatmaps, depth_maps, control_pack, keypoints_dict, pose3d_dict
                del frames_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            chunk_elapsed = time.time() - chunk_start
            self.stats["chunk_times"].append(chunk_elapsed)
            self.stats["total_chunks"] += 1

            logger.info(
                f"    分片 {chunk_idx + 1} 完成: {chunk_elapsed:.1f}s "
                f"({n_frames / chunk_elapsed:.1f} fps)"
            )

        # 清理模型缓存
        clear_model_cache()
        self.stats["stage_times"]["02-06_chunks"] = time.time() - t0

    # ════════════════════════════════════════════════════════════
    # 阶段 3: 全局后处理
    # ════════════════════════════════════════════════════════════

    def _stage_post_process(self) -> str:
        """07→09 全局后处理: 合并所有分片 → 光流矫正 → 融合 → 成片"""
        t0 = time.time()
        logger.info("[阶段 3/3] 全局后处理: 合并 → 07 RAFT → 08 Fusion → 09 Composite")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── 合并所有分片 ──
        logger.info(f"  合并 {len(self.all_ltx_frames)} 个分片...")
        original_all = torch.cat(self.all_original_frames, dim=0)  # (T, 3, H, W)
        ltx_all = torch.cat(self.all_ltx_frames, dim=0)
        masks_all = torch.cat(self.all_masks, dim=0)

        total_frames = original_all.shape[0]
        logger.info(f"  合并完成: {total_frames} 帧, {original_all.shape}")

        # 释放 CPU 内存中的分片列表
        del self.all_ltx_frames, self.all_original_frames, self.all_masks

        # ── 07 RAFT 光流矫正 ──
        t = time.time()
        original_gpu = original_all.to(device)
        ltx_gpu = ltx_all.to(device)
        masks_gpu = masks_all.to(device)

        aligned = raft_correct(original_gpu, ltx_gpu, masks_gpu)
        self.stats["stage_times"]["07_raft"] = time.time() - t
        logger.info(f"  07 RAFT: {aligned.shape} ({(time.time()-t):.1f}s)")

        del ltx_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 08 分层光影融合 ──
        t = time.time()
        final = fusion_blend(aligned, masks_gpu, depth_maps=None)  # depth_maps 可选
        self.stats["stage_times"]["08_fusion"] = time.time() - t
        logger.info(f"  08 Fusion: {final.shape} ({(time.time()-t):.1f}s)")

        # ── 09 FFmpeg 成片 ──
        t = time.time()
        output_path = os.path.join(self.output_dir, "output.mp4")
        result = composite_video(final, self.audio_path, self.metadata, output_path)
        self.stats["stage_times"]["09_composite"] = time.time() - t
        logger.info(f"  09 Composite: {result} ({(time.time()-t):.1f}s)")

        # 清理
        del original_gpu, aligned, final, masks_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.stats["stage_times"]["07-09_post"] = time.time() - t0
        return result


# ── 便捷函数 ──

def run_pipeline(
    video_path: str,
    click_points: list = None,
    reference_dir: str = "",
    prompt: str = "",
    **kwargs,
) -> str:
    """一键运行全流程"""
    runner = PipelineRunner(
        video_path=video_path,
        click_points=click_points,
        reference_dir=reference_dir,
        prompt=prompt,
        **kwargs,
    )
    return runner.run()