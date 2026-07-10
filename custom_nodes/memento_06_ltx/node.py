"""Memento 06 — LTX-Video 2.3 + IC-LoRA 多特征融合渲染节点

基于原生 ltx-pipelines ICLoraPipeline（非 diffusers）。
IC-LoRA 通过 LoraPathStrengthAndSDOps 原生加载，无需 peft。
控制信号：Pose 骨架视频 + Depth 深度视频 + Canny 边缘视频 → MP4。
"""
import logging
import json
import os
import time
from pathlib import Path

import torch
import cv2
import numpy as np

from .control_extractor import (
    generate_pose_video,
    generate_depth_video,
    generate_canny_video,
    align_resolution,
    align_frames,
)

logger = logging.getLogger(__name__)

# ── LTX-2 原生导入（Dockerfile 中预安装） ──
try:
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization.policy import QuantizationPolicy
    _LTX_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[MementoLTX] LTX-2 原生管线未安装: {e}")
    _LTX_AVAILABLE = False


class MementoLTX:
    """节点 6: 多特征融合渲染 — LTX-Video 2.3 + IC-LoRA 原生控制

    使用 LTX-2 原生 ICLoraPipeline：
    - 主模型: /models/ltx/ltx-2.3-22b-dev-fp8.safetensors
    - IC-LoRA: /models/iclora/ 下的 pose/depth/canny 适配器
    - 控制信号: Pose/Depth/Canny 三个 MP4 视频
    - 显存: FP8 量化约 10GB（主模型）+ IC-LoRA 开销
    """

    # 模型路径
    MAIN_MODEL = "/models/ltx/ltx-2.3-22b-dev-fp8.safetensors"
    ICLORA_POSE = "/models/iclora/ltx-video-iclora-pose-13b-0.9.7.safetensors"
    ICLORA_DEPTH = "/models/iclora/ltx-video-iclora-depth-13b-0.9.7.safetensors"
    ICLORA_CANNY = "/models/iclora/ltx-video-iclora-canny-13b-0.9.7.safetensors"

    _pipeline = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": ("STRING", {"default": "", "multiline": False}),
                "quadmask_dir": ("STRING", {"default": "", "multiline": False}),
                "pose3d_json_path": ("STRING", {"default": "", "multiline": False}),
                "mask_dir": ("STRING", {"default": "", "multiline": False}),
                "control_mode": (
                    ["pose", "depth", "canny", "pose+depth", "pose+canny", "depth+canny", "pose+depth+canny"],
                    {"default": "pose+depth+canny"}
                ),
                "control_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "num_inference_steps": ("INT", {"default": 8, "min": 1, "max": 50}),
                "prompt": ("STRING", {"default": "cinematic, high quality, realistic", "multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("synthetic_dir",)
    FUNCTION = "generate"
    CATEGORY = "Memento/06_LTX"

    @classmethod
    def load_pipeline(cls, control_mode: str, control_strength: float):
        """懒加载 ICLoraPipeline，缓存单例"""
        if cls._pipeline is not None:
            return cls._pipeline

        if not _LTX_AVAILABLE:
            raise ImportError(
                "LTX-2 原生管线未安装。请在 Dockerfile 中确保:\n"
                "  git clone https://github.com/Lightricks/LTX-2.git /opt/ltx2\n"
                "  cd /opt/ltx2 && pip install -e packages/ltx-core -e packages/ltx-pipelines"
            )

        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 检查主模型
        if not os.path.exists(cls.MAIN_MODEL):
            raise FileNotFoundError(
                f"LTX-Video 2.3 主模型不存在: {cls.MAIN_MODEL}\n"
                f"请先运行 bash download_models.sh 下载模型"
            )

        # 构建 IC-LoRA 列表
        loras = []
        if "pose" in control_mode:
            if os.path.exists(cls.ICLORA_POSE):
                loras.append(LoraPathStrengthAndSDOps(
                    path=cls.ICLORA_POSE, strength=control_strength
                ))
            else:
                logger.warning(f"[MementoLTX] Pose IC-LoRA 不存在: {cls.ICLORA_POSE}")

        if "depth" in control_mode:
            if os.path.exists(cls.ICLORA_DEPTH):
                loras.append(LoraPathStrengthAndSDOps(
                    path=cls.ICLORA_DEPTH, strength=control_strength
                ))
            else:
                logger.warning(f"[MementoLTX] Depth IC-LoRA 不存在: {cls.ICLORA_DEPTH}")

        if "canny" in control_mode:
            if os.path.exists(cls.ICLORA_CANNY):
                loras.append(LoraPathStrengthAndSDOps(
                    path=cls.ICLORA_CANNY, strength=control_strength
                ))
            else:
                logger.warning(f"[MementoLTX] Canny IC-LoRA 不存在: {cls.ICLORA_CANNY}")

        if not loras:
            raise RuntimeError(f"没有可用的 IC-LoRA 模型，control_mode={control_mode}")

        logger.info(f"[MementoLTX] 加载 LTX-Video 2.3 + {len(loras)} IC-LoRA...")

        # 初始化 ICLoraPipeline
        pipeline = ICLoraPipeline(
            distilled_checkpoint_path=cls.MAIN_MODEL,
            loras=loras,
            device=device,
            quantization=QuantizationPolicy.fp8_cast(),
        )

        elapsed = time.time() - start_time
        logger.info(f"[MementoLTX] 管线加载完成，耗时 {elapsed:.1f}s")

        # 显存检查
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoLTX] 显存占用: {mem_used:.2f} GB")
            if mem_used > 10.5:
                logger.warning(
                    f"[MementoLTX] 显存占用超过 10.5GB 限制: {mem_used:.2f} GB"
                )

        cls._pipeline = pipeline
        return pipeline

    def get_frame_info(self, frames_dir: str) -> tuple[int, int, int]:
        """获取帧序列的尺寸和帧数"""
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        h, w = first.shape[:2]
        return h, w, len(frame_files)

    def generate(self, frames_dir: str, quadmask_dir: str, pose3d_json_path: str,
                 mask_dir: str, control_mode: str, control_strength: float,
                 num_inference_steps: int, prompt: str, seed: int):
        logger.info(
            f"[MementoLTX] frames={frames_dir}, mode={control_mode}, "
            f"strength={control_strength}, steps={num_inference_steps}"
        )

        # 参数验证
        for path, name in [
            (frames_dir, "frames_dir"),
            (quadmask_dir, "quadmask_dir"),
            (pose3d_json_path, "pose3d_json_path"),
            (mask_dir, "mask_dir"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} 不存在: {path}")

        # 获取帧信息
        h, w, num_frames = self.get_frame_info(frames_dir)
        logger.info(f"[MementoLTX] 原始: {num_frames} 帧, {w}x{h}")

        # 对齐分辨率和帧数到 LTX-Video 要求
        h_a, w_a = align_resolution(h, w)
        num_frames_a = align_frames(num_frames, "8n+1")
        if h != h_a or w != w_a:
            logger.info(f"[MementoLTX] 分辨率对齐: {w}x{h} → {w_a}x{h_a}")
        if num_frames != num_frames_a:
            logger.info(f"[MementoLTX] 帧数对齐: {num_frames} → {num_frames_a}")

        # 创建输出目录
        synthetic_dir = "/workspace/synthetic"
        controls_dir = "/workspace/controls"
        Path(synthetic_dir).mkdir(parents=True, exist_ok=True)
        Path(controls_dir).mkdir(parents=True, exist_ok=True)

        # ── 生成控制信号视频 ──
        fps = 25
        control_videos = []

        if "pose" in control_mode:
            pose_mp4 = os.path.join(controls_dir, "pose_control.mp4")
            generate_pose_video(pose3d_json_path, pose_mp4, h_a, w_a, num_frames_a, fps)
            control_videos.append((pose_mp4, control_strength))

        if "depth" in control_mode:
            depth_mp4 = os.path.join(controls_dir, "depth_control.mp4")
            generate_depth_video(mask_dir, depth_mp4, h_a, w_a, num_frames_a, fps)
            control_videos.append((depth_mp4, control_strength))

        if "canny" in control_mode:
            canny_mp4 = os.path.join(controls_dir, "canny_control.mp4")
            generate_canny_video(frames_dir, mask_dir, canny_mp4, h_a, w_a, num_frames_a, fps)
            control_videos.append((canny_mp4, control_strength))

        logger.info(f"[MementoLTX] 控制信号: {len(control_videos)} 个视频")

        # ── 加载管线 + 推理 ──
        pipeline = self.load_pipeline(control_mode, control_strength)

        logger.info(f"[MementoLTX] 开始推理 (seed={seed}, steps={num_inference_steps})...")
        infer_start = time.time()

        try:
            # 调用 ICLoraPipeline
            video_iterator, _ = pipeline(
                prompt=prompt,
                seed=seed,
                height=h_a,
                width=w_a,
                num_frames=num_frames_a,
                frame_rate=fps,
                conditioning_attention_strength=control_strength,
                video_conditioning=control_videos,
                enhance_prompt=True,
                skip_stage_2=False,
            )

            # 保存生成帧
            saved_count = 0
            for frame_idx in range(min(num_frames, num_frames_a)):
                try:
                    frame = next(video_iterator)
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).clip(0, 255).astype(np.uint8)

                    # 如果帧尺寸不对，resize
                    if frame.shape[:2] != (h, w):
                        frame = cv2.resize(frame, (w, h))

                    # BGR → RGB
                    if frame.shape[-1] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    out_path = os.path.join(synthetic_dir, f"synth_{frame_idx+1:05d}.png")
                    cv2.imwrite(out_path, frame)
                    saved_count += 1
                except StopIteration:
                    break
                except Exception as e:
                    logger.warning(f"[MementoLTX] 帧 {frame_idx+1} 保存失败: {e}")

        except Exception as e:
            logger.error(f"[MementoLTX] 推理失败: {e}")
            raise RuntimeError(f"LTX-Video 推理失败: {e}") from e

        infer_elapsed = time.time() - infer_start
        fps_render = saved_count / infer_elapsed if infer_elapsed > 0 else 0
        logger.info(
            f"[MementoLTX] 推理完成: {saved_count} 帧, "
            f"耗时 {infer_elapsed:.1f}s ({fps_render:.2f} fps)"
        )

        # 更新 context.json
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "synthetic_dir": synthetic_dir,
            "num_synthetic_frames": saved_count,
            "control_mode": control_mode,
            "control_strength": control_strength,
            "inference_time_sec": round(infer_elapsed, 1),
            "inference_fps": round(fps_render, 2),
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        # 显存报告
        if torch.cuda.is_available():
            mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoLTX] 峰值显存: {mem_peak:.2f} GB")

        logger.info(f"[MementoLTX] 合成帧输出到 {synthetic_dir}")
        return (synthetic_dir,)


NODE_CLASS_MAPPINGS = {"MementoLTX": MementoLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoLTX": "Memento 06 - LTX-Video 2.3 + IC-LoRA 多特征融合渲染"}