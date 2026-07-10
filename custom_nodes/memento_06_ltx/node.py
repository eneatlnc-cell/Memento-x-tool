"""Memento 06 — LTX-Video 2.3 + IC-LoRA 局部重绘（Inpainting）节点

基于原生 ltx-pipelines ICLoraPipeline（非 diffusers）。
IC-LoRA 通过 LoraPathStrengthAndSDOps 原生加载，无需 peft。
控制信号：Pose 热力图 + Depth 深度图 + Canny/Distance/Temporal 控制包 → MP4。

核心行为：
- 背景保持原始不变（通过 mask 合成）
- 仅 mask 区域内的人物被替换为角色 B
- 动作、姿态、空间透视、边缘轮廓由 IC-LoRA 约束
- 合成公式：生成人物区域 + 原始背景 = 最终帧
"""

import json
import logging
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .control_extractor import (
    align_frames,
    align_resolution,
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


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

def images_dir_to_mp4(
    image_dir: str,
    output_mp4: str,
    h: int,
    w: int,
    num_frames: int,
    fps: int = 30,
    image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
) -> str:
    """将控制信号图像序列目录转换为 MP4 视频，作为 LTX IC-LoRA 的 video_conditioning 输入。

    处理流程：
    1. 扫描目录中所有图像文件并排序
    2. 逐帧缩放至目标分辨率 (w, h)
    3. 超出范围的帧用空白帧填充
    4. 通过 FFmpeg 合成 H.264 MP4

    Args:
        image_dir: 控制信号帧目录（预计算的 Pose/Depth/Canny 帧）
        output_mp4: 输出 MP4 文件路径
        h: 目标高度（已对齐至 64 的倍数）
        w: 目标宽度（已对齐至 64 的倍数）
        num_frames: 目标帧数（已对齐至 8n+1）
        fps: 帧率
        image_extensions: 支持的图像扩展名

    Returns:
        输出 MP4 路径

    Raises:
        FileNotFoundError: 目录不存在
        RuntimeError: 目录为空或 FFmpeg 合成失败
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"控制信号目录不存在: {image_dir}")

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        raise RuntimeError(
            f"控制信号目录为空（无有效图像文件）: {image_dir}"
        )

    logger.info(
        f"[MementoLTX] 图像序列 → MP4: {Path(image_dir).name} "
        f"({len(image_files)} 个文件, 目标 {num_frames} 帧, {w}x{h}, {fps}fps)"
    )

    # 创建临时帧目录
    tmp_dir = str(Path(output_mp4).parent / f".tmp_{Path(output_mp4).stem}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        for i in range(num_frames):
            if i < len(image_files):
                img_path = os.path.join(image_dir, image_files[i])
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(
                        f"[MementoLTX] 无法读取控制帧: {img_path}, 使用空白帧代替"
                    )
                    img = np.zeros((h, w, 3), dtype=np.uint8)
                else:
                    if img.shape[:2] != (h, w):
                        img = cv2.resize(img, (w, h))
            else:
                # 目录帧数不足，用空白帧填充
                img = np.zeros((h, w, 3), dtype=np.uint8)

            cv2.imwrite(os.path.join(tmp_dir, f"frame_{i+1:05d}.png"), img)

        # FFmpeg 合成 MP4
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_mp4,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg 合成控制视频失败 ({Path(output_mp4).name}): "
                f"{result.stderr[:500]}"
            )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"[MementoLTX] 控制视频已生成: {output_mp4}")
    return output_mp4


def load_reference_images(reference_dir: str) -> List[Image.Image]:
    """从目录加载角色 B 参考图像（所有 .jpg/.png 文件）。

    预期目录结构（5 视角参考）：
      - front_face.png       (正面脸部)
      - side_45.png          (45度侧面)
      - side_standard.png    (标准侧面)
      - back.png             (背面)
      - full_body_3view.png  (全身三视图)

    实际加载时不对文件名做假设，仅加载所有支持格式的图像。

    Args:
        reference_dir: 参考图像目录路径

    Returns:
        PIL Image 列表（RGB 模式）

    Raises:
        FileNotFoundError: 目录不存在
        RuntimeError: 目录为空或所有图像加载失败
    """
    if not os.path.isdir(reference_dir):
        raise FileNotFoundError(f"参考图像目录不存在: {reference_dir}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    image_files = sorted([
        f for f in os.listdir(reference_dir)
        if f.lower().endswith(image_extensions)
    ])

    if not image_files:
        raise RuntimeError(
            f"参考图像目录为空（无 .jpg/.png/.webp 文件）: {reference_dir}\n"
            f"请确保目录中包含 5 个视角的角色 B 参考图: "
            f"正面脸部、45度侧面、标准侧面、背面、全身三视图"
        )

    reference_images = []
    for fname in image_files:
        img_path = os.path.join(reference_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
            reference_images.append(img)
            logger.info(
                f"[MementoLTX] 已加载参考图像: {fname} ({img.size[0]}x{img.size[1]})"
            )
        except Exception as e:
            logger.warning(
                f"[MementoLTX] 无法加载参考图像 {fname}: {e}, 跳过"
            )

    if not reference_images:
        raise RuntimeError(
            f"未能成功加载任何参考图像: {reference_dir}"
        )

    logger.info(
        f"[MementoLTX] 共加载 {len(reference_images)} 张角色 B 参考图像"
    )
    return reference_images


def load_mask_frame(mask_path: str, h: int, w: int) -> np.ndarray:
    """加载单帧掩码并缩放到目标尺寸，输出二值掩码。

    Args:
        mask_path: 掩码 PNG 文件路径
        h: 目标高度
        w: 目标宽度

    Returns:
        二值掩码 (h, w) dtype=uint8, 值 0（背景）或 255（人物区域）
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(
            f"[MementoLTX] 无法读取掩码: {mask_path}, 使用全零掩码（无重绘区域）"
        )
        return np.zeros((h, w), dtype=np.uint8)

    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 二值化：阈值 127
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def composite_frame(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """合成最终帧：生成人物区域 + 原始背景。

    使用掩码进行 alpha 混合：
      result = generated * mask_float + original * (1 - mask_float)

    其中 mask_float 是归一化到 [0, 1] 的三通道掩码。

    Args:
        original: 原始帧 (H, W, 3) BGR uint8 — 背景保持原样
        generated: LTX 生成帧 (H, W, 3) BGR uint8 — 含新角色
        mask: 二值掩码 (H, W) uint8, 255=人物区域（需替换）, 0=背景（保持）

    Returns:
        合成帧 (H, W, 3) BGR uint8
    """
    h, w = original.shape[:2]

    # 确保所有输入尺寸一致
    if generated.shape[:2] != (h, w):
        generated = cv2.resize(generated, (w, h))
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 转换为 float 掩码 [0, 1]，扩展为三通道
    mask_float = mask.astype(np.float32) / 255.0
    mask_3ch = np.stack([mask_float] * 3, axis=-1)

    # Alpha 混合
    original_f = original.astype(np.float32)
    generated_f = generated.astype(np.float32)
    composited_f = generated_f * mask_3ch + original_f * (1.0 - mask_3ch)

    return np.clip(composited_f, 0, 255).astype(np.uint8)


def parse_metadata(metadata_json: str) -> Dict:
    """解析原始视频元数据 JSON 文件。

    期望格式：
    {
        "fps": 30,
        "width": 1920,   (或 "w")
        "height": 1080,  (或 "h")
        "duration": 10.5
    }

    Args:
        metadata_json: metadata.json 文件路径

    Returns:
        包含 fps, width, height, duration 的字典，缺失字段使用默认值
    """
    defaults = {
        "fps": 30,
        "width": 1920,
        "height": 1080,
        "duration": 0.0,
    }

    if not metadata_json or not os.path.isfile(metadata_json):
        logger.warning(
            f"[MementoLTX] metadata.json 不存在: {metadata_json}, "
            f"使用默认值: {defaults}"
        )
        return defaults

    try:
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(
            f"[MementoLTX] metadata.json 解析失败: {e}, 使用默认值"
        )
        return defaults

    result = {
        "fps": int(metadata.get("fps", defaults["fps"])),
        "width": int(metadata.get("width", metadata.get("w", defaults["width"]))),
        "height": int(metadata.get("height", metadata.get("h", defaults["height"]))),
        "duration": float(metadata.get("duration", defaults["duration"])),
    }

    logger.info(
        f"[MementoLTX] Metadata: fps={result['fps']}, "
        f"{result['width']}x{result['height']}, "
        f"duration={result['duration']:.1f}s"
    )
    return result


def prepare_reference_tensor(
    ref_image: Image.Image,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> torch.Tensor:
    """将 PIL 参考图像转换为 LTX 管线可接受的张量格式。

    缩放至目标分辨率，归一化到 [0, 1]，转换为 [1, 3, H, W] 格式。

    Args:
        ref_image: PIL 参考图像
        target_h: 目标高度
        target_w: 目标宽度
        device: torch 设备

    Returns:
        (1, 3, H, W) float32 张量，值域 [0, 1]
    """
    ref_resized = ref_image.resize((target_w, target_h), Image.LANCZOS)
    ref_np = np.array(ref_resized).astype(np.float32) / 255.0
    ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0)
    return ref_tensor.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# 主节点类
# ═══════════════════════════════════════════════════════════════════════════════

class MementoLTX:
    """节点 6: 局部重绘 — LTX-Video 2.3 + IC-LoRA 原生控制

    使用 LTX-2 原生 ICLoraPipeline：
    - 主模型: /models/ltx/ltx-2.3-22b-dev-fp8.safetensors
    - IC-LoRA: /models/iclora/ 下的 pose/depth/canny 适配器
    - 控制信号: Pose/Depth/Canny 三个预计算 MP4 视频
    - 参考图像: 角色 B 的 5 视角参考图
    - 显存: FP8 量化约 10GB（主模型）+ IC-LoRA 开销

    核心行为：
    - 背景保持原始不变（通过 mask 合成实现）
    - 仅 mask 区域（人物）被 LTX 生成的角色 B 替换
    - 动作、姿态、空间透视、边缘轮廓由 IC-LoRA 约束
    """

    # ── 模型路径 ──
    MAIN_MODEL = "/models/ltx/ltx-2.3-22b-dev-fp8.safetensors"
    ICLORA_POSE = "/models/iclora/ltx-video-iclora-pose-13b-0.9.7.safetensors"
    ICLORA_DEPTH = "/models/iclora/ltx-video-iclora-depth-13b-0.9.7.safetensors"
    ICLORA_CANNY = "/models/iclora/ltx-video-iclora-canny-13b-0.9.7.safetensors"

    # 管线缓存: key=(control_mode, control_strength) → pipeline 实例
    # 不同控制模式需要不同的 IC-LoRA 组合，因此按组合缓存
    _pipeline_cache: Dict[Tuple[str, float], object] = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "01_frames_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "原始 30fps 帧目录（背景保持不变）",
                }),
                "02_mask_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "重绘掩码目录（仅 mask 区域内的人物被替换）",
                }),
                "03_heatmap_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Pose 控制: 姿态热力图帧目录（预计算）",
                }),
                "04_depth_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Depth 控制: 深度图帧目录（预计算）",
                }),
                "05_control_pack_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Canny/Distance/Temporal 控制包帧目录（预计算）",
                }),
                "06_reference_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "角色 B 参考图像目录（5 视角: 正面/45度侧/标准侧/背面/全身三视图）",
                }),
                "07_prompt": ("STRING", {
                    "default": "cinematic, high quality, realistic, detailed character, professional lighting",
                    "multiline": True,
                    "tooltip": "角色 B 文本提示词（描述要生成的角色外观）",
                }),
                "08_metadata_json": ("STRING", {
                    "default": "/workspace/metadata.json",
                    "multiline": False,
                    "tooltip": "原始视频元数据 JSON（fps, width, height, duration）",
                }),
                "control_mode": (
                    [
                        "pose",
                        "depth",
                        "canny",
                        "pose+depth",
                        "pose+canny",
                        "depth+canny",
                        "pose+depth+canny",
                    ],
                    {"default": "pose+depth+canny"},
                ),
                "control_strength": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "IC-LoRA 控制强度（越高越严格遵循控制信号）",
                }),
                "num_inference_steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "去噪步数（8 步为推荐值，步数越多质量越高但速度越慢）",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "随机种子（固定种子可复现结果）",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("synthetic_dir",)
    FUNCTION = "generate"
    CATEGORY = "Memento/06_LTX"

    # ═══════════════════════════════════════════════════════════════════════════
    # 管线加载（懒加载 + 缓存）
    # ═══════════════════════════════════════════════════════════════════════════

    @classmethod
    def load_pipeline(cls, control_mode: str, control_strength: float):
        """懒加载 ICLoraPipeline，按 (control_mode, control_strength) 缓存。

        不同控制模式需要不同的 IC-LoRA 组合，因此按组合键缓存管线实例。
        同一 (mode, strength) 组合的多次调用将复用同一管线，避免重复加载。

        Args:
            control_mode: 控制模式字符串（如 "pose+depth+canny"）
            control_strength: IC-LoRA 控制强度

        Returns:
            ICLoraPipeline 实例
        """
        cache_key = (control_mode, control_strength)
        if cache_key in cls._pipeline_cache:
            logger.info(
                f"[MementoLTX] 复用已缓存的管线: mode={control_mode}, "
                f"strength={control_strength}"
            )
            return cls._pipeline_cache[cache_key]

        if not _LTX_AVAILABLE:
            raise ImportError(
                "LTX-2 原生管线未安装。请在 Dockerfile 中确保:\n"
                "  git clone https://github.com/Lightricks/LTX-2.git /opt/ltx2\n"
                "  cd /opt/ltx2 && pip install -e packages/ltx-core "
                "-e packages/ltx-pipelines"
            )

        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            logger.info(
                f"[MementoLTX] GPU: {gpu_name}, 总显存: {total_mem:.1f} GB"
            )
        else:
            logger.warning(
                "[MementoLTX] CUDA 不可用，将使用 CPU 推理（速度极慢，不推荐）"
            )

        # 检查主模型
        if not os.path.exists(cls.MAIN_MODEL):
            raise FileNotFoundError(
                f"LTX-Video 2.3 主模型不存在: {cls.MAIN_MODEL}\n"
                f"请先运行 bash download_models.sh 下载模型"
            )

        # 构建 IC-LoRA 列表
        iclora_map = {
            "pose": cls.ICLORA_POSE,
            "depth": cls.ICLORA_DEPTH,
            "canny": cls.ICLORA_CANNY,
        }

        loras = []
        for mode_key, lora_path in iclora_map.items():
            if mode_key in control_mode:
                if os.path.exists(lora_path):
                    loras.append(
                        LoraPathStrengthAndSDOps(
                            path=lora_path,
                            strength=control_strength,
                        )
                    )
                    logger.info(
                        f"[MementoLTX] 已添加 IC-LoRA: {mode_key} "
                        f"(strength={control_strength}, path={Path(lora_path).name})"
                    )
                else:
                    logger.warning(
                        f"[MementoLTX] IC-LoRA 模型不存在: {lora_path}, "
                        f"跳过 {mode_key} 控制"
                    )

        if not loras:
            raise RuntimeError(
                f"没有可用的 IC-LoRA 模型，control_mode={control_mode}\n"
                f"请检查 /models/iclora/ 目录下的模型文件是否完整"
            )

        logger.info(
            f"[MementoLTX] 加载 LTX-Video 2.3 + {len(loras)} 个 IC-LoRA: "
            f"{control_mode}"
        )

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
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoLTX] 显存占用: {mem_used:.2f} GB")
            if mem_used > 10.5:
                logger.warning(
                    f"[MementoLTX] 显存占用超过 10.5GB: {mem_used:.2f} GB, "
                    f"请考虑降低分辨率或减少控制信号数量"
                )

        cls._pipeline_cache[cache_key] = pipeline
        return pipeline

    # ═══════════════════════════════════════════════════════════════════════════
    # 帧信息获取
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def get_frame_info(frames_dir: str) -> Tuple[int, int, int]:
        """获取帧序列的尺寸和帧数。

        通过读取第一帧获取分辨率，统计目录中所有图像文件获取帧数。

        Args:
            frames_dir: 帧目录路径

        Returns:
            (高度, 宽度, 帧数) 三元组

        Raises:
            RuntimeError: 目录为空或无法读取第一帧
        """
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise RuntimeError(f"帧目录为空: {frames_dir}")

        first_path = os.path.join(frames_dir, frame_files[0])
        first = cv2.imread(first_path)
        if first is None:
            raise RuntimeError(f"无法读取第一帧: {first_path}")

        h, w = first.shape[:2]
        return h, w, len(frame_files)

    # ═══════════════════════════════════════════════════════════════════════════
    # 主生成函数
    # ═══════════════════════════════════════════════════════════════════════════

    def generate(
        self,
        frames_dir: str,           # 01_frames_dir
        mask_dir: str,             # 02_mask_dir
        heatmap_dir: str,          # 03_heatmap_dir
        depth_dir: str,            # 04_depth_dir
        control_pack_dir: str,     # 05_control_pack_dir
        reference_dir: str,        # 06_reference_dir
        prompt: str,               # 07_prompt
        metadata_json: str,        # 08_metadata_json
        control_mode: str,
        control_strength: float,
        num_inference_steps: int,
        seed: int,
    ):
        """执行局部重绘生成。

        完整流程：
        1. 验证所有输入路径的存在性
        2. 解析 metadata.json 获取原始视频参数（fps, 分辨率, 时长）
        3. 获取帧目录的实际尺寸和帧数
        4. 加载角色 B 参考图像（5 视角）
        5. 对齐分辨率（64 的倍数）和帧数（8n+1）到 LTX-Video 要求
        6. 将三个控制信号目录（Pose/Depth/Canny）转换为 MP4 视频
        7. 加载 ICLoraPipeline 并执行 LTX 推理
        8. 逐帧合成：生成人物区域 + 原始背景（mask 控制）
        9. 保存合成帧到 synthetic_dir
        10. 更新 context.json 记录生成参数和结果

        Returns:
            (synthetic_dir,) — 合成帧输出目录的路径字符串
        """
        total_start = time.time()
        logger.info("=" * 60)
        logger.info("[MementoLTX] ====== 局部重绘开始 ======")
        logger.info(f"  帧目录:       {frames_dir}")
        logger.info(f"  掩码目录:     {mask_dir}")
        logger.info(f"  热力图目录:   {heatmap_dir}")
        logger.info(f"  深度图目录:   {depth_dir}")
        logger.info(f"  控制包目录:   {control_pack_dir}")
        logger.info(f"  参考图目录:   {reference_dir}")
        logger.info(f"  控制模式:     {control_mode}")
        logger.info(f"  控制强度:     {control_strength}")
        logger.info(f"  推理步数:     {num_inference_steps}")
        logger.info(f"  随机种子:     {seed}")
        logger.info(f"  提示词:       {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
        logger.info("=" * 60)

        # ── 步骤 1: 验证所有输入路径 ──
        required_paths = {
            "01_frames_dir": frames_dir,
            "02_mask_dir": mask_dir,
            "03_heatmap_dir": heatmap_dir,
            "04_depth_dir": depth_dir,
            "05_control_pack_dir": control_pack_dir,
            "06_reference_dir": reference_dir,
        }

        for name, path in required_paths.items():
            if not path or not path.strip():
                raise ValueError(
                    f"[MementoLTX] {name} 未设置，请提供有效的目录路径"
                )
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[MementoLTX] {name} 不存在: {path}"
                )

        # ── 步骤 2: 解析 metadata.json ──
        metadata = parse_metadata(metadata_json)
        original_fps = metadata["fps"]
        original_w = metadata["width"]
        original_h = metadata["height"]

        # ── 步骤 3: 获取帧目录实际信息 ──
        h, w, num_frames = self.get_frame_info(frames_dir)
        logger.info(
            f"[MementoLTX] 帧目录: {num_frames} 帧, {w}x{h}, "
            f"metadata: {original_w}x{original_h}, {original_fps}fps"
        )

        # 以 metadata 中的分辨率为准（如果帧目录分辨率不一致，使用 metadata 值）
        if w != original_w or h != original_h:
            logger.warning(
                f"[MementoLTX] 帧目录分辨率 ({w}x{h}) 与 metadata "
                f"({original_w}x{original_h}) 不一致，以 metadata 为准"
            )
            w, h = original_w, original_h

        # ── 步骤 4: 加载角色参考图像 ──
        reference_images = load_reference_images(reference_dir)

        # ── 步骤 5: 对齐分辨率和帧数 ──
        h_a, w_a = align_resolution(h, w)
        num_frames_a = align_frames(num_frames, "8n+1")
        if h != h_a or w != w_a:
            logger.info(
                f"[MementoLTX] 分辨率对齐: {w}x{h} → {w_a}x{h_a} "
                f"(64 的倍数)"
            )
        if num_frames != num_frames_a:
            logger.info(
                f"[MementoLTX] 帧数对齐: {num_frames} → {num_frames_a} "
                f"(8n+1 格式)"
            )

        # ── 步骤 6: 创建输出目录 ──
        synthetic_dir = "/workspace/synthetic"
        controls_dir = "/workspace/controls"
        Path(synthetic_dir).mkdir(parents=True, exist_ok=True)
        Path(controls_dir).mkdir(parents=True, exist_ok=True)

        # 清空之前的合成帧（避免新旧帧混合）
        for old_file in Path(synthetic_dir).glob("synth_*.png"):
            old_file.unlink()

        # ── 步骤 7: 控制信号目录 → MP4 视频 ──
        control_dir_map = {
            "pose": heatmap_dir,
            "depth": depth_dir,
            "canny": control_pack_dir,
        }
        control_name_map = {
            "pose": "pose_control",
            "depth": "depth_control",
            "canny": "canny_control",
        }

        control_videos = []
        for mode_key, ctrl_dir in control_dir_map.items():
            if mode_key not in control_mode:
                continue

            mp4_name = f"{control_name_map[mode_key]}.mp4"
            mp4_path = os.path.join(controls_dir, mp4_name)

            try:
                images_dir_to_mp4(
                    image_dir=ctrl_dir,
                    output_mp4=mp4_path,
                    h=h_a,
                    w=w_a,
                    num_frames=num_frames_a,
                    fps=original_fps,
                )
                control_videos.append((mp4_path, control_strength))
                logger.info(
                    f"[MementoLTX] 控制信号 [{mode_key}] 就绪: {mp4_path}"
                )
            except Exception as e:
                logger.error(
                    f"[MementoLTX] 控制信号 [{mode_key}] 转换失败: {e}"
                )
                raise RuntimeError(
                    f"控制信号 [{mode_key}] 转换失败: {e}"
                ) from e

        if not control_videos:
            raise RuntimeError(
                f"没有成功转换任何控制信号视频，control_mode={control_mode}"
            )

        logger.info(
            f"[MementoLTX] 共 {len(control_videos)} 个控制信号视频准备就绪"
        )

        # ── 步骤 8: 加载管线 + 执行 LTX 推理 ──
        pipeline = self.load_pipeline(control_mode, control_strength)

        logger.info(
            f"[MementoLTX] 开始推理 (seed={seed}, steps={num_inference_steps}, "
            f"frames={num_frames_a}, {w_a}x{h_a}, {original_fps}fps)..."
        )
        infer_start = time.time()

        # 准备参考图像张量
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_tensors = []
        for ref_img in reference_images:
            ref_tensor = prepare_reference_tensor(ref_img, h_a, w_a, device)
            ref_tensors.append(ref_tensor)

        # 构建管线参数
        pipeline_kwargs = {
            "prompt": prompt,
            "seed": seed,
            "height": h_a,
            "width": w_a,
            "num_frames": num_frames_a,
            "frame_rate": original_fps,
            "conditioning_attention_strength": control_strength,
            "video_conditioning": control_videos,
            "enhance_prompt": True,
            "skip_stage_2": False,
        }

        # 尝试传递参考图像（如果 ICLoraPipeline 支持 image 参数）
        if ref_tensors:
            pipeline_kwargs["image"] = ref_tensors[0]
            logger.info(
                f"[MementoLTX] 使用参考图像作为条件: "
                f"{reference_images[0].size[0]}x{reference_images[0].size[1]}"
            )

        try:
            video_iterator, _ = pipeline(**pipeline_kwargs)
        except TypeError as type_err:
            if "image" in str(type_err):
                logger.warning(
                    "[MementoLTX] ICLoraPipeline 不支持 'image' 参数，"
                    "将不使用参考图像进行推理"
                )
                pipeline_kwargs.pop("image", None)
                video_iterator, _ = pipeline(**pipeline_kwargs)
            else:
                raise

        # ── 步骤 9: 逐帧合成（生成人物 + 原始背景） ──
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith('.png')
        ])

        if not mask_files:
            raise RuntimeError(f"掩码目录为空（无 .png 文件）: {mask_dir}")

        logger.info(
            f"[MementoLTX] 开始逐帧合成: {num_frames} 帧..."
        )

        saved_count = 0
        effective_frames = min(num_frames, num_frames_a)

        for frame_idx in range(effective_frames):
            # ── 获取生成的帧 ──
            try:
                gen_frame = next(video_iterator)
            except StopIteration:
                logger.warning(
                    f"[MementoLTX] 生成帧提前结束于第 {frame_idx + 1} 帧 "
                    f"(预期 {effective_frames} 帧)"
                )
                break
            except Exception as e:
                logger.error(
                    f"[MementoLTX] 获取生成帧 {frame_idx + 1} 失败: {e}"
                )
                break

            # 张量 → numpy 数组
            if isinstance(gen_frame, torch.Tensor):
                gen_frame = gen_frame.detach().cpu().numpy()
            if gen_frame.dtype != np.uint8:
                gen_frame = (gen_frame * 255).clip(0, 255).astype(np.uint8)

            # 如果生成帧尺寸与对齐分辨率不同，缩放回原始分辨率
            if gen_frame.shape[:2] != (h_a, w_a):
                gen_frame = cv2.resize(gen_frame, (w_a, h_a))
            if gen_frame.shape[:2] != (h, w):
                gen_frame = cv2.resize(gen_frame, (w, h))

            # 通道顺序处理：LTX 输出通常为 RGB，需转为 BGR 供 OpenCV 保存
            if gen_frame.ndim == 3 and gen_frame.shape[-1] == 3:
                gen_frame = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR)

            # ── 加载原始帧 ──
            orig_path = os.path.join(frames_dir, frame_files[frame_idx])
            orig_frame = cv2.imread(orig_path)
            if orig_frame is None:
                logger.warning(
                    f"[MementoLTX] 无法读取原始帧: {orig_path}, 跳过"
                )
                continue
            if orig_frame.shape[:2] != (h, w):
                orig_frame = cv2.resize(orig_frame, (w, h))

            # ── 加载掩码 ──
            mask_idx = frame_idx if frame_idx < len(mask_files) else len(mask_files) - 1
            mask_path = os.path.join(mask_dir, mask_files[mask_idx])
            mask = load_mask_frame(mask_path, h, w)

            # ── 合成：生成人物区域 + 原始背景 ──
            composited = composite_frame(
                original=orig_frame,
                generated=gen_frame,
                mask=mask,
            )

            # ── 保存合成帧 ──
            out_path = os.path.join(synthetic_dir, f"synth_{frame_idx + 1:05d}.png")
            cv2.imwrite(out_path, composited)
            saved_count += 1

            # 进度日志（每 10 帧或首帧）
            if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                # 计算掩码覆盖率用于调试
                mask_coverage = mask.sum() / (h * w * 255) * 100
                logger.info(
                    f"[MementoLTX] 合成进度: {frame_idx + 1}/{effective_frames} "
                    f"(掩码覆盖率: {mask_coverage:.1f}%)"
                )

        infer_elapsed = time.time() - infer_start
        total_elapsed = time.time() - total_start
        fps_render = saved_count / infer_elapsed if infer_elapsed > 0 else 0

        logger.info(
            f"[MementoLTX] 推理完成: {saved_count}/{effective_frames} 帧合成成功, "
            f"推理耗时 {infer_elapsed:.1f}s ({fps_render:.2f} fps), "
            f"总耗时 {total_elapsed:.1f}s"
        )

        if saved_count == 0:
            raise RuntimeError(
                "[MementoLTX] 未生成任何合成帧，请检查输入数据和管线配置"
            )

        # ── 步骤 10: 更新 context.json ──
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            try:
                with open(context_path, "r") as f:
                    context = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(
                    f"[MementoLTX] context.json 读取失败: {e}, 将创建新文件"
                )

        context.update({
            "synthetic_dir": synthetic_dir,
            "num_synthetic_frames": saved_count,
            "control_mode": control_mode,
            "control_strength": control_strength,
            "num_reference_images": len(reference_images),
            "reference_dir": reference_dir,
            "inference_time_sec": round(infer_elapsed, 1),
            "inference_fps": round(fps_render, 2),
            "total_time_sec": round(total_elapsed, 1),
            "original_resolution": f"{w}x{h}",
            "aligned_resolution": f"{w_a}x{h_a}",
            "original_fps": original_fps,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "inpainting_mode": True,
            "prompt": prompt,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        logger.info(f"[MementoLTX] context.json 已更新: {context_path}")

        # ── 步骤 11: 显存报告 ──
        if torch.cuda.is_available():
            mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            logger.info(
                f"[MementoLTX] 峰值显存: allocated={mem_peak:.2f} GB, "
                f"reserved={mem_reserved:.2f} GB"
            )

        logger.info(
            f"[MementoLTX] ====== 局部重绘完成! ======\n"
            f"  合成帧输出: {synthetic_dir}\n"
            f"  生成帧数:   {saved_count}\n"
            f"  背景策略:   保持原始（通过 mask 合成）\n"
            f"  控制模式:   {control_mode}\n"
            f"  参考图像:   {len(reference_images)} 张"
        )

        return (synthetic_dir,)


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI 节点注册
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {"MementoLTX": MementoLTX}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MementoLTX": "Memento 06 - LTX-Video 2.3 + IC-LoRA 局部重绘 (Inpainting)"
}