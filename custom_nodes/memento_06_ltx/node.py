"""Memento 06 — LTX-Video 2.3 + IC-LoRA 局部重绘（Inpainting）节点

基于原生 ltx-pipelines ICLoraPipeline（非 diffusers）。
IC-LoRA 通过 LoraPathStrengthAndSDOps 原生加载，无需 peft。

双 LoRA 约束架构:
  - Union Control (depth+canny+pose 三合一) → 结构控制
  - Ingredients (Reference Sheet) → 角色外观一致性

控制信号来源（8路输入）:
  - 01_frames_dir      → 来自 01 原始 30fps 帧
  - 02_mask_dir        → 来自 02 SAM3 视频分割（人物 Mask）
  - 03_heatmap_dir     → 来自 03 MediaPipe 2D 骨骼热力图（Pose 控制）
  - 04_depth_dir       → 来自 04 MotionBERT 深度图（Depth 控制）
  - 05_control_pack_dir → 来自 05 对齐控制（Canny/Distance/Temporal 控制包）
  - 06_reference_dir   → 角色 B 参考图像（5 视角）
  - 07_prompt          → 文本提示词（描述角色 B 外观）
  - 08_metadata_json   → 原始视频元数据（fps, width, height, duration）

核心行为：
- 背景保持原始不变（通过 mask 合成）
- 仅 mask 区域内的人物被替换为角色 B
- 动作、姿态、空间透视、边缘轮廓由 Union Control IC-LoRA 约束
- 角色外观一致性由 Ingredients IC-LoRA Reference Sheet 约束
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

# ── 尝试导入 memento_pipeline.ops.sub GPU 张量操作 ──
try:
    from memento_pipeline.ops.sub import ltx_inpaint as _ops_ltx_inpaint
    _TENSOR_OPS_AVAILABLE = True
    logger.info("[MementoLTX] memento_pipeline.ops.sub 已加载，将使用 GPU 张量操作")
except ImportError:
    _TENSOR_OPS_AVAILABLE = False
    logger.info("[MementoLTX] memento_pipeline.ops.sub 未安装，使用文件级回退逻辑")

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
# 控制信号来源文档
# ═══════════════════════════════════════════════════════════════════════════════

CONTROL_SOURCES = (
    "Mask            → 来自 02 SAM3 视频分割 (人物区域)\n"
    "Pose (Heatmap)  → 来自 03 MediaPipe 2D 骨骼热力图\n"
    "Depth           → 来自 04 MotionBERT 深度图\n"
    "Canny/Distance  → 来自 05 对齐控制 (Canny边缘 + Distance距离图)\n"
    "Temporal        → 来自 05 时序平滑参数\n"
    "Reference       → 角色 B 参考图像 (5视角)\n"
    "Prompt          → 文本提示词 (描述角色 B 外观)\n"
    "Metadata        → 原始视频元数据 (fps, width, height)"
)


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


def build_reference_sheet(
    reference_images: List[Image.Image],
    sheet_width: int = 1920,
    sheet_height: int = 1080,
    columns: int = 5,
) -> Image.Image:
    """将多张参考图像合成为一张 Reference Sheet（黑底 + 多视角布局）。

    Ingredients IC-LoRA 要求输入一个 Reference Sheet：一张在黑色背景上
    展示角色、道具、场景的合成图像。模型通过该 Sheet 保持角色外观一致性。

    布局策略：横向排列，每张参考图缩放到等宽等高的单元格内。
    如果参考图数量不足，剩余单元格留空（黑色）。

    Args:
        reference_images: PIL Image 列表（角色参考图，5 视角）
        sheet_width: Reference Sheet 总宽度
        sheet_height: Reference Sheet 总高度
        columns: 列数（默认 5 列，对应 5 视角）

    Returns:
        Reference Sheet PIL Image (RGB, 黑色背景)
    """
    n = len(reference_images)
    if n == 0:
        raise ValueError("参考图像列表为空，无法生成 Reference Sheet")

    rows = (n + columns - 1) // columns
    cell_w = sheet_width // columns
    cell_h = sheet_height // rows

    # 创建黑色背景画布
    canvas = Image.new("RGB", (sheet_width, sheet_height), (0, 0, 0))

    for idx, img in enumerate(reference_images):
        row = idx // columns
        col = idx % columns

        # 缩放到单元格内（保持宽高比，填充整个单元格）
        img_ratio = img.width / img.height
        cell_ratio = cell_w / cell_h

        if img_ratio > cell_ratio:
            # 图像更宽 → 按高度缩放
            new_h = cell_h
            new_w = int(cell_h * img_ratio)
        else:
            # 图像更高 → 按宽度缩放
            new_w = cell_w
            new_h = int(cell_w / img_ratio)

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)

        # 居中放置
        x_offset = col * cell_w + (cell_w - new_w) // 2
        y_offset = row * cell_h + (cell_h - new_h) // 2

        canvas.paste(img_resized, (x_offset, y_offset))

    logger.info(
        f"[MementoLTX] Reference Sheet 生成: {n} 张参考图 → "
        f"{columns}x{rows} 布局, {sheet_width}x{sheet_height}"
    )
    return canvas


def reference_sheet_to_static_video(
    reference_sheet: Image.Image,
    output_mp4: str,
    num_frames: int,
    fps: int = 24,
) -> str:
    """将 Reference Sheet 转换为静态视频（同一帧循环）。

    Ingredients IC-LoRA 要求 Reference Sheet 以静态视频形式输入
    （同一帧循环到输出片段长度），最少 121 帧。

    Args:
        reference_sheet: Reference Sheet PIL Image
        output_mp4: 输出 MP4 路径
        num_frames: 目标帧数（最低 121 帧）
        fps: 帧率

    Returns:
        输出 MP4 路径
    """
    # Ingredients 最低要求 121 帧
    effective_frames = max(num_frames, 121)

    tmp_dir = str(Path(output_mp4).parent / f".refsheet_tmp_{Path(output_mp4).stem}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # 将 Reference Sheet 转为 numpy，写入一帧
        sheet_np = np.array(reference_sheet)
        # RGB → BGR for OpenCV
        sheet_bgr = cv2.cvtColor(sheet_np, cv2.COLOR_RGB2BGR)

        single_frame_path = os.path.join(tmp_dir, "frame_00001.png")
        cv2.imwrite(single_frame_path, sheet_bgr)

        # 用 FFmpeg loop 生成静态视频
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", single_frame_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-t", str(effective_frames / fps),
            "-r", str(fps),
            output_mp4,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg Reference Sheet 静态视频生成失败: {result.stderr[:500]}"
            )

        logger.info(
            f"[MementoLTX] Reference Sheet 静态视频: {output_mp4} "
            f"({effective_frames} 帧, {fps}fps)"
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return output_mp4


def build_ingredients_prompt(user_prompt: str) -> str:
    """构建 Ingredients 双段式 Prompt。

    Ingredients IC-LoRA 要求 Prompt 格式为：
      "Reference sheet: <面板描述> / Generated video: <动作描述>"

    Args:
        user_prompt: 用户输入的原始 Prompt（描述角色 B 外观和动作）

    Returns:
        Ingredients 格式的 Prompt
    """
    # 自动生成 Reference sheet 面板描述
    ref_description = (
        "A character reference sheet with multiple viewing angles "
        "showing the same character: front face close-up, 45-degree side view, "
        "standard side profile, back view, and full body three-view. "
        "The character maintains consistent facial features, hairstyle, "
        "body type, and costume across all panels."
    )

    return f"Reference sheet: {ref_description} / Generated video: {user_prompt}"


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
    """节点 6: 局部重绘 — LTX-Video 2.3 + IC-LoRA 双 LoRA 约束 (8路输入)

    双 LoRA 架构:
      - Union Control (depth+canny+pose 三合一) → 结构控制
      - Ingredients (Reference Sheet) → 角色外观一致性

    8路输入信号:
      - 01 原始帧 (背景保持)
      - 02 Mask (来自 SAM3)
      - 03 Pose 热力图 (来自 MediaPipe)
      - 04 Depth 深度图 (来自 MotionBERT)
      - 05 Canny/Distance/Temporal 控制包 (来自 05 对齐)
      - 06 角色 B 参考图像 (5 视角)
      - 07 文本提示词
      - 08 元数据 JSON

    使用 LTX-2 原生 ICLoraPipeline：
    - 主模型: /models/ltx/ltx-2.3-22b-dev-fp8.safetensors
    - IC-LoRA Union: /models/iclora/ 下 Union Control 三合一 (depth+canny+pose)
    - IC-LoRA Ingredients: /models/iclora/ 下 Reference Sheet 角色一致性
    - 控制信号: Pose/Depth/Canny 三个预计算 MP4 视频
    - 参考图像: 角色 B 的 5 视角参考图 → 自动合成 Reference Sheet
    - 显存: FP8 量化约 10GB（主模型）+ Union Control 654MB + Ingredients

    核心行为：
    - 背景保持原始不变（通过 mask 合成实现）
    - 仅 mask 区域（人物）被 LTX 生成的角色 B 替换
    - 动作、姿态、空间透视、边缘轮廓由 Union Control IC-LoRA 约束
    - 角色外观（面部、服装、体型）由 Ingredients Reference Sheet 约束
    """

    # ── 模型路径 ──
    MAIN_MODEL = "/models/ltx/ltx-2.3-22b-dev-fp8.safetensors"
    ICLORA_UNION = "/models/iclora/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
    ICLORA_INGREDIENTS = "/models/iclora/ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors"

    # 管线缓存: key=(control_mode, control_strength, ingredients_enabled, ingredients_strength) → pipeline 实例
    _pipeline_cache: Dict[Tuple[str, float, bool, float], object] = {}

    # 是否使用 GPU 张量操作
    _use_tensor_ops = _TENSOR_OPS_AVAILABLE

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
                    "tooltip": "Union Control 控制强度（越高越严格遵循控制信号）",
                }),
                "ingredients_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用 Ingredients Reference Sheet 角色外观一致性约束",
                }),
                "ingredients_strength": ("FLOAT", {
                    "default": 1.4,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Ingredients 控制强度（推荐 1.4，越高角色外观越严格匹配参考图）",
                }),
                "num_inference_steps": ("INT", {
                    "default": 30,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "去噪步数（Ingredients 模式下推荐 30 步，普通模式 8 步）",
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**32 - 1,
                    "step": 1,
                    "tooltip": "随机种子（固定种子可复现结果）",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                    "tooltip": "CFG 引导强度，3-5=修复推荐，7-9=创意生成",
                }),
                "stg_scale": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "STG 时空引导，3.0=动态场景推荐，1.5=静态场景",
                }),
            },
            "hidden": {
                "control_sources": ("STRING", {
                    "default": CONTROL_SOURCES,
                    "multiline": True,
                    "tooltip": "8路控制信号来源说明（文档用途）",
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
    def load_pipeline(cls, control_mode: str, control_strength: float,
                      ingredients_enabled: bool = True, ingredients_strength: float = 1.4):
        """懒加载 ICLoraPipeline，按 (control_mode, control_strength, ingredients_enabled, ingredients_strength) 缓存。

        双 LoRA 架构：
        - Union Control (depth+canny+pose 三合一) → 结构控制
        - Ingredients (Reference Sheet) → 角色外观一致性

        Args:
            control_mode: 控制模式字符串（如 "pose+depth+canny"）
            control_strength: Union Control 控制强度
            ingredients_enabled: 是否启用 Ingredients Reference Sheet 约束
            ingredients_strength: Ingredients 控制强度 (推荐 1.4)

        Returns:
            ICLoraPipeline 实例
        """
        cache_key = (control_mode, control_strength, ingredients_enabled, ingredients_strength)
        if cache_key in cls._pipeline_cache:
            logger.info(
                f"[MementoLTX] 复用已缓存的管线: mode={control_mode}, "
                f"strength={control_strength}, ingredients={'on' if ingredients_enabled else 'off'}"
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
        loras = []

        # ── Union Control (depth+canny+pose 三合一) ──
        if not os.path.exists(cls.ICLORA_UNION):
            raise FileNotFoundError(
                f"IC-LoRA Union Control 模型不存在: {cls.ICLORA_UNION}\n"
                f"请先运行 bash download_models.sh 下载模型"
            )
        loras.append(
            LoraPathStrengthAndSDOps(
                path=cls.ICLORA_UNION,
                strength=control_strength,
            )
        )
        logger.info(
            f"[MementoLTX] Union Control: strength={control_strength}"
        )

        # ── Ingredients (Reference Sheet 角色一致性) ──
        if ingredients_enabled:
            if not os.path.exists(cls.ICLORA_INGREDIENTS):
                logger.warning(
                    f"[MementoLTX] Ingredients IC-LoRA 不存在: {cls.ICLORA_INGREDIENTS}, "
                    f"将跳过 Ingredients 约束"
                )
            else:
                loras.append(
                    LoraPathStrengthAndSDOps(
                        path=cls.ICLORA_INGREDIENTS,
                        strength=ingredients_strength,
                    )
                )
                logger.info(
                    f"[MementoLTX] Ingredients: strength={ingredients_strength}"
                )

        logger.info(
            f"[MementoLTX] 加载 LTX-Video 2.3 + {len(loras)} 个 IC-LoRA: "
            f"Union Control{' + Ingredients' if ingredients_enabled else ''} "
            f"(mode={control_mode})"
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
    # GPU 张量操作路径
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_tensor_ops(self, frames_dir, mask_dir, heatmap_dir, depth_dir,
                              control_pack_dir, reference_dir, prompt, metadata_json,
                              control_mode, control_strength, ingredients_enabled,
                              ingredients_strength, num_inference_steps, seed):
        """使用 memento_pipeline.ops.sub.ltx_inpaint 进行 GPU 张量操作"""
        if not _TENSOR_OPS_AVAILABLE:
            raise RuntimeError("memento_pipeline.ops.sub 不可用，无法使用张量操作路径")

        total_start = time.time()
        logger.info("=" * 60)
        logger.info("[MementoLTX] ====== 局部重绘开始 (tensor ops) ======")

        # 验证所有输入路径
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
                raise ValueError(f"[MementoLTX] {name} 未设置，请提供有效的目录路径")
            if not os.path.exists(path):
                raise FileNotFoundError(f"[MementoLTX] {name} 不存在: {path}")

        metadata = parse_metadata(metadata_json)
        h, w, num_frames = self.get_frame_info(frames_dir)

        # 加载帧为张量 (N, 3, H, W) float32 [0, 1]
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        frames_np = []
        for fn in frame_files[:num_frames]:
            img = cv2.imread(os.path.join(frames_dir, fn))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_np.append(img)
        frames_np = np.stack(frames_np, axis=0)
        frames_t = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)

        # 加载掩码为张量 (N, 1, H, W) float32 [0, 1]
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith('.png')
        ])
        masks_np = np.zeros((num_frames, h, w), dtype=np.float32)
        for i in range(min(num_frames, len(mask_files))):
            m = cv2.imread(os.path.join(mask_dir, mask_files[i]), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h))
                masks_np[i] = m.astype(np.float32) / 255.0
        masks_t = torch.from_numpy(masks_np).unsqueeze(1)

        # 加载控制包为张量 (N, 4, H, W) float32 [0, 1]
        control_files = sorted([
            f for f in os.listdir(control_pack_dir)
            if f.lower().endswith('.png')
        ])
        control_np = np.zeros((num_frames, 4, h, w), dtype=np.float32)
        for i in range(min(num_frames, len(control_files))):
            cp = cv2.imread(os.path.join(control_pack_dir, control_files[i]), cv2.IMREAD_UNCHANGED)
            if cp is not None:
                if cp.shape[:2] != (h, w):
                    cp = cv2.resize(cp, (w, h))
                if cp.ndim == 2:
                    cp = cp[:, :, None]
                for c in range(min(4, cp.shape[2])):
                    control_np[i, c] = cp[:, :, c].astype(np.float32) / 255.0
        control_t = torch.from_numpy(control_np)

        # 设置 metadata dict 包含 control_mode 和 ingredients 参数
        metadata_dict = {
            "fps": metadata["fps"],
            "width": w,
            "height": h,
            "control_mode": control_mode,
            "ingredients_enabled": ingredients_enabled,
            "ingredients_strength": ingredients_strength,
        }

        # 调用 ops.sub.ltx_inpaint
        synthetic_t = _ops_ltx_inpaint(
            frames=frames_t,
            masks=masks_t,
            control_pack=control_t,
            reference_dir=reference_dir,
            prompt=prompt,
            metadata=metadata_dict,
            control_strength=control_strength,
            ingredients_enabled=ingredients_enabled,
            ingredients_strength=ingredients_strength,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )  # (N, 3, H, W) float32 [0, 1]

        # 创建输出目录
        synthetic_dir = "/workspace/synthetic"
        Path(synthetic_dir).mkdir(parents=True, exist_ok=True)
        for old_file in Path(synthetic_dir).glob("synth_*.png"):
            old_file.unlink()

        # 保存合成帧
        synthetic_np = synthetic_t.cpu().numpy()  # (N, 3, H, W)
        saved_count = 0
        for i in range(num_frames):
            frame = synthetic_np[i].transpose(1, 2, 0)  # (H, W, 3)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out_path = os.path.join(synthetic_dir, f"synth_{i+1:05d}.png")
            cv2.imwrite(out_path, frame)
            saved_count += 1

        total_elapsed = time.time() - total_start
        logger.info(
            f"[MementoLTX] (tensor ops) 完成: {saved_count} 帧合成成功, "
            f"总耗时 {total_elapsed:.1f}s"
        )

        # 更新 context.json
        self._update_context(synthetic_dir, saved_count, control_mode, control_strength,
                             reference_dir, num_inference_steps, seed, w, h, prompt, metadata, total_elapsed)
        return (synthetic_dir,)

    # ═══════════════════════════════════════════════════════════════════════════
    # 文件级回退路径
    # ═══════════════════════════════════════════════════════════════════════════

    def _generate_file_based(
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
        ingredients_enabled: bool,
        ingredients_strength: float,
        num_inference_steps: int,
        seed: int,
    ):
        """执行局部重绘生成（文件级回退路径）。

        完整流程：
        1. 验证所有输入路径的存在性
        2. 解析 metadata.json 获取原始视频参数（fps, 分辨率, 时长）
        3. 获取帧目录的实际尺寸和帧数
        4. 加载角色 B 参考图像（5 视角）
        5. [Ingredients] 生成 Reference Sheet + 静态视频
        6. 对齐分辨率（64 的倍数）和帧数（8n+1）到 LTX-Video 要求
        7. 将三个控制信号目录（Pose/Depth/Canny）转换为 MP4 视频
        8. 加载双 LoRA 管线并执行 LTX 推理
        9. 逐帧合成：生成人物区域 + 原始背景（mask 控制）
        10. 保存合成帧到 synthetic_dir
        11. 更新 context.json 记录生成参数和结果

        Returns:
            (synthetic_dir,) — 合成帧输出目录的路径字符串
        """
        total_start = time.time()
        logger.info("=" * 60)
        logger.info("[MementoLTX] ====== 局部重绘开始 (file-based) ======")
        logger.info(f"  帧目录:       {frames_dir}")
        logger.info(f"  掩码目录:     {mask_dir}")
        logger.info(f"  热力图目录:   {heatmap_dir}")
        logger.info(f"  深度图目录:   {depth_dir}")
        logger.info(f"  控制包目录:   {control_pack_dir}")
        logger.info(f"  参考图目录:   {reference_dir}")
        logger.info(f"  控制模式:     {control_mode}")
        logger.info(f"  Union强度:    {control_strength}")
        logger.info(f"  Ingredients:  {'启用' if ingredients_enabled else '禁用'} (strength={ingredients_strength})")
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

        # ── 步骤 5: [Ingredients] 生成 Reference Sheet + 静态视频 ──
        refsheet_video_path = None
        if ingredients_enabled:
            logger.info("[MementoLTX] 生成 Ingredients Reference Sheet...")
            try:
                ref_sheet = build_reference_sheet(
                    reference_images,
                    sheet_width=1920,
                    sheet_height=1080,
                )
                refsheet_video_path = os.path.join(
                    "/workspace", "controls", "refsheet_ingredients.mp4"
                )
                os.makedirs(os.path.dirname(refsheet_video_path), exist_ok=True)
                reference_sheet_to_static_video(
                    ref_sheet,
                    refsheet_video_path,
                    num_frames,
                    fps=original_fps,
                )
                logger.info(
                    f"[MementoLTX] Ingredients Reference Sheet 视频就绪: {refsheet_video_path}"
                )
            except Exception as e:
                logger.warning(
                    f"[MementoLTX] Reference Sheet 生成失败: {e}, "
                    f"将禁用 Ingredients 约束"
                )
                ingredients_enabled = False

        # ── 步骤 6: 对齐分辨率和帧数 ──
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

        # ── 步骤 7: 创建输出目录 ──
        synthetic_dir = "/workspace/synthetic"
        controls_dir = "/workspace/controls"
        Path(synthetic_dir).mkdir(parents=True, exist_ok=True)
        Path(controls_dir).mkdir(parents=True, exist_ok=True)

        # 清空之前的合成帧（避免新旧帧混合）
        for old_file in Path(synthetic_dir).glob("synth_*.png"):
            old_file.unlink()

        # ── 步骤 8: 控制信号目录 → MP4 视频 ──
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

        # 添加 Ingredients Reference Sheet 视频
        if ingredients_enabled and refsheet_video_path:
            control_videos.append((refsheet_video_path, ingredients_strength))
            logger.info(
                f"[MementoLTX] Ingredients Reference Sheet 已添加到控制信号: "
                f"strength={ingredients_strength}"
            )

        logger.info(
            f"[MementoLTX] 共 {len(control_videos)} 个控制信号视频准备就绪"
        )

        # ── 步骤 9: 构建 Ingredients Prompt ──
        if ingredients_enabled:
            effective_prompt = build_ingredients_prompt(prompt)
            logger.info(
                f"[MementoLTX] Ingredients Prompt: "
                f"{effective_prompt[:200]}..."
            )
        else:
            effective_prompt = prompt

        # ── 步骤 10: 加载管线 + 执行 LTX 推理 ──
        pipeline = self.load_pipeline(
            control_mode, control_strength,
            ingredients_enabled, ingredients_strength,
        )

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
            "prompt": effective_prompt,
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

        # Ingredients 模式下的负向 Prompt
        if ingredients_enabled:
            pipeline_kwargs["negative_prompt"] = (
                "worst quality, inconsistent motion, blurry, jittery, distorted"
            )

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
            elif "negative_prompt" in str(type_err):
                logger.warning(
                    "[MementoLTX] ICLoraPipeline 不支持 'negative_prompt' 参数"
                )
                pipeline_kwargs.pop("negative_prompt", None)
                video_iterator, _ = pipeline(**pipeline_kwargs)
            else:
                raise

        # ── 步骤 11: 逐帧合成（生成人物 + 原始背景） ──
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

        # ── 步骤 12: 更新 context.json ──
        self._update_context(synthetic_dir, saved_count, control_mode, control_strength,
                             reference_dir, num_inference_steps, seed, w, h, prompt, metadata, total_elapsed)

        logger.info(
            f"[MementoLTX] ====== 局部重绘完成! ======\n"
            f"  合成帧输出: {synthetic_dir}\n"
            f"  生成帧数:   {saved_count}\n"
            f"  背景策略:   保持原始（通过 mask 合成）\n"
            f"  控制模式:   {control_mode}\n"
            f"  Ingredients: {'启用' if ingredients_enabled else '禁用'}\n"
            f"  参考图像:   {len(reference_images)} 张"
        )

        return (synthetic_dir,)

    # ═══════════════════════════════════════════════════════════════════════════
    # context.json 更新
    # ═══════════════════════════════════════════════════════════════════════════

    def _update_context(self, synthetic_dir, saved_count, control_mode, control_strength,
                        reference_dir, num_inference_steps, seed, w, h, prompt, metadata, total_elapsed):
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
            "num_reference_images": len(os.listdir(reference_dir)) if os.path.isdir(reference_dir) else 0,
            "reference_dir": reference_dir,
            "inference_time_sec": round(total_elapsed, 1),
            "total_time_sec": round(total_elapsed, 1),
            "original_resolution": f"{w}x{h}",
            "original_fps": metadata["fps"],
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "inpainting_mode": True,
            "prompt": prompt,
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        logger.info(f"[MementoLTX] context.json 已更新: {context_path}")

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
        ingredients_enabled: bool,
        ingredients_strength: float,
        num_inference_steps: int,
        seed: int,
        control_sources: str = "",  # hidden parameter — documentation only
    ):
        """执行局部重绘生成。

        优先使用 memento_pipeline.ops.sub.ltx_inpaint GPU 张量操作，
        如果不可用或失败则回退到文件级逻辑。

        Returns:
            (synthetic_dir,) — 合成帧输出目录的路径字符串
        """
        logger.info(
            f"[MementoLTX] generate: tensor_ops={self._use_tensor_ops}, "
            f"mode={control_mode}, strength={control_strength}, "
            f"ingredients={'on' if ingredients_enabled else 'off'}"
        )

        if self._use_tensor_ops and _TENSOR_OPS_AVAILABLE:
            try:
                return self._generate_tensor_ops(
                    frames_dir, mask_dir, heatmap_dir, depth_dir,
                    control_pack_dir, reference_dir, prompt, metadata_json,
                    control_mode, control_strength, ingredients_enabled,
                    ingredients_strength, num_inference_steps, seed,
                )
            except Exception as e:
                logger.warning(
                    f"[MementoLTX] GPU 张量操作失败: {e}，回退到文件级逻辑"
                )
                logger.debug(traceback.format_exc())

        return self._generate_file_based(
            frames_dir, mask_dir, heatmap_dir, depth_dir,
            control_pack_dir, reference_dir, prompt, metadata_json,
            control_mode, control_strength, ingredients_enabled,
            ingredients_strength, num_inference_steps, seed,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI 节点注册
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {"MementoLTX": MementoLTX}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MementoLTX": "Memento 06 - LTX 局部重绘 (8路输入)"
}