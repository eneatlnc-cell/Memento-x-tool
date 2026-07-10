"""Memento Pipeline Ops -- GPU Tensor-based Operations for Nodes 06-09
=====================================================================

All functions take torch.Tensor as input and return torch.Tensor as output
(except composite_video which returns a file path string).  No file I/O for
intermediate steps -- everything happens on the GPU.

Nodes:
  06  ltx_inpaint        LTX-Video 2.3 + IC-LoRA  local inpainting
  07  raft_correct        RAFT optical-flow temporal correction
  08  fusion_blend        4-layer lighting/colour fusion
  09  composite_video     FFmpeg final encode
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# =============================================================================
# Module-level MODEL_CACHE  (singletons for LTX pipeline, RAFT model)
# =============================================================================
MODEL_CACHE: Dict[str, object] = {
    "_ltx_pipeline": None,          # ICLoraPipeline instance
    "_ltx_control_key": None,       # (control_mode, control_strength) cache key
    "_raft_model": None,            # RAFT torchvision model
    "_raft_device": None,           # torch.device for RAFT
}

# =============================================================================
# Optional-import guards
# =============================================================================
_LTX_AVAILABLE = False
try:
    from ltx_pipelines.ic_lora import ICLoraPipeline
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization.policy import QuantizationPolicy
    _LTX_AVAILABLE = True
except ImportError:
    pass

_RAFT_AVAILABLE = False
try:
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    _RAFT_AVAILABLE = True
except ImportError:
    pass

_OPENCV_AVAILABLE = True
try:
    import cv2
except ImportError:
    _OPENCV_AVAILABLE = False
    logger.warning("[sub] OpenCV not available; some fallback paths will fail.")


# =============================================================================
# Internal helpers
# =============================================================================

def _to_float32(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is float32 in [0,1]."""
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    elif x.dtype != torch.float32:
        x = x.float()
    return x.clamp(0.0, 1.0)


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 [0,1] tensor to uint8 [0,255]."""
    return (x.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 2D Gaussian kernel tensor (1, 1, K, K)."""
    ax = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _apply_gaussian_blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a batch of images (N, C, H, W) or single image (C, H, W)."""
    was_single = (x.dim() == 3)
    if was_single:
        x = x.unsqueeze(0)
    device = x.device
    kernel = _gaussian_kernel_2d(kernel_size, sigma, device)
    # Pad to preserve size
    pad = kernel_size // 2
    # Group convolution: one filter per channel
    C = x.shape[1]
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(x_padded, kernel, groups=C)
    if was_single:
        blurred = blurred.squeeze(0)
    return blurred


def _warp_tensor(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp image using optical flow via grid_sample.

    Args:
        img: (1, C, H, W) or (C, H, W) float32
        flow: (1, 2, H, W) or (2, H, W) float32, channels = (dx, dy) in pixel units

    Returns:
        warped: same shape as img
    """
    if img.dim() == 3:
        img = img.unsqueeze(0)
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)

    N, C, H, W = img.shape
    # Build normalised grid
    gy, gx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=torch.float32),
        torch.arange(W, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N, H, W, 2)

    # flow: (N, 2, H, W) -> (N, H, W, 2)
    flow_perm = flow.permute(0, 2, 3, 1)
    vgrid = grid + flow_perm

    # Normalise to [-1, 1]
    vgrid[..., 0] = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid[..., 1] = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0

    warped = F.grid_sample(img, vgrid, mode="bilinear", padding_mode="border", align_corners=True)
    if warped.shape[0] == 1 and img.shape[0] == 1:
        warped = warped.squeeze(0)
    return warped


def _flow_magnitude(flow: torch.Tensor) -> torch.Tensor:
    """Compute per-pixel flow magnitude.  flow: (..., 2, H, W) -> (..., H, W)."""
    return torch.sqrt(flow[..., 0, :, :] ** 2 + flow[..., 1, :, :] ** 2)


def _tensor_to_numpy_frame(x: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float32 [0,1] tensor -> (H, W, 3) uint8 numpy BGR."""
    x = x.detach().cpu()
    if x.dtype != torch.float32:
        x = x.float()
    x = x.clamp(0.0, 1.0)
    # CHW -> HWC
    np_img = (x.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    # RGB -> BGR (OpenCV convention)
    if np_img.shape[-1] == 3:
        np_img = np_img[..., ::-1].copy()
    return np_img


def _tensor_to_numpy_gray(x: torch.Tensor) -> np.ndarray:
    """Convert (1, H, W) or (H, W) float32 [0,1] tensor -> (H, W) uint8 numpy."""
    x = x.detach().cpu()
    if x.dtype != torch.float32:
        x = x.float()
    x = x.clamp(0.0, 1.0)
    if x.dim() == 3:
        x = x.squeeze(0)
    return (x.numpy() * 255.0).astype(np.uint8)


def _numpy_frame_to_tensor(np_img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W, 3) uint8 numpy BGR -> (3, H, W) float32 [0,1] tensor."""
    if np_img.shape[-1] == 3:
        np_img = np_img[..., ::-1].copy()  # BGR -> RGB
    tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)
    if tensor.dim() == 3:
        tensor = tensor.permute(2, 0, 1)
    return tensor.to(device)


def _numpy_gray_to_tensor(np_img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (H, W) uint8 numpy -> (1, H, W) float32 [0,1] tensor."""
    tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    return tensor.to(device)


# =============================================================================
# 06 -- LTX Local Inpainting
# =============================================================================

def _build_control_mp4_from_channel(
    channel_tensor: torch.Tensor,   # (N, H, W) float32 [0,1]
    output_mp4: str,
    fps: int,
    tag: str,
) -> str:
    """Write a single-channel control tensor to an MP4 video file.

    This is the only file I/O in node 06 -- required because the IC-LoRA
    interface accepts video files, not tensors.
    """
    if not _OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV is required for control MP4 generation.")

    N, H, W = channel_tensor.shape
    tmp_dir = tempfile.mkdtemp(prefix=f"memento_control_{tag}_")
    try:
        for i in range(N):
            frame = (channel_tensor[i].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            # Expand to 3-channel for video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{i + 1:05d}.png"), frame_rgb)

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
                f"FFmpeg control MP4 [{tag}] failed: {result.stderr[:500]}"
            )
        logger.info(f"[sub.06] Control MP4 [{tag}] ready: {output_mp4} ({N} frames, {W}x{H})")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_mp4


def _load_ltx_pipeline(
    control_mode: str,
    control_strength: float,
    main_model: str = "/models/ltx/ltx-2.3-22b-dev-fp8.safetensors",
    iclora_dir: str = "/models/iclora/",
) -> object:
    """Load or return cached ICLoraPipeline singleton."""
    cache_key = (control_mode, control_strength)
    cached = MODEL_CACHE.get("_ltx_pipeline")
    cached_key = MODEL_CACHE.get("_ltx_control_key")
    if cached is not None and cached_key == cache_key:
        logger.info(f"[sub.06] Reusing cached LTX pipeline (mode={control_mode}, strength={control_strength})")
        return cached

    if not _LTX_AVAILABLE:
        raise ImportError(
            "LTX-2 native pipelines not installed.  Ensure ltx-pipelines and ltx-core "
            "are available in the Docker image."
        )

    if not os.path.exists(main_model):
        raise FileNotFoundError(f"LTX main model not found: {main_model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        logger.info(f"[sub.06] GPU: {gpu_name}, total VRAM: {total_mem:.1f} GB")

    iclora_map = {
        "pose":  os.path.join(iclora_dir, "ltx-video-iclora-pose-13b-0.9.7.safetensors"),
        "depth": os.path.join(iclora_dir, "ltx-video-iclora-depth-13b-0.9.7.safetensors"),
        "canny": os.path.join(iclora_dir, "ltx-video-iclora-canny-13b-0.9.7.safetensors"),
    }

    loras = []
    for mode_key, lora_path in iclora_map.items():
        if mode_key in control_mode:
            if os.path.exists(lora_path):
                loras.append(
                    LoraPathStrengthAndSDOps(path=lora_path, strength=control_strength)
                )
                logger.info(f"[sub.06] IC-LoRA added: {mode_key} (strength={control_strength})")
            else:
                logger.warning(f"[sub.06] IC-LoRA not found: {lora_path}, skipping {mode_key}")

    if not loras:
        raise RuntimeError(f"No IC-LoRA models available for control_mode={control_mode}")

    start = time.time()
    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=main_model,
        loras=loras,
        device=device,
        quantization=QuantizationPolicy.fp8_cast(),
    )
    elapsed = time.time() - start
    logger.info(f"[sub.06] LTX pipeline loaded in {elapsed:.1f}s")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info(f"[sub.06] VRAM after load: {mem_used:.2f} GB")

    MODEL_CACHE["_ltx_pipeline"] = pipeline
    MODEL_CACHE["_ltx_control_key"] = cache_key
    return pipeline


def ltx_inpaint(
    frames: torch.Tensor,           # (N, 3, H, W) float32 [0,1]
    masks: torch.Tensor,            # (N, 1, H, W) float32 [0,1]
    control_pack: torch.Tensor,     # (N, 4, H, W) float32 [0,1]
    reference_dir: str,             # path to character-B reference images
    prompt: str,                    # text prompt
    metadata: dict,                 # {"fps": 30, "width": 1920, "height": 1080, ...}
    control_strength: float = 0.7,
    num_inference_steps: int = 8,
    seed: int = 42,
) -> torch.Tensor:
    """Node 06: LTX-Video 2.3 + IC-LoRA local inpainting.

    Control signals from control_pack channels:
      channel 0 (R) = Canny     (from node 05)
      channel 1 (G) = Distance  (from node 05)
      channel 2 (B) = Pose      (from node 03, via 05)
      channel 3 (A) = Temporal  (from node 05)

    Composite:  result = generated_person * mask + original_frame * (1 - mask)
    Background is preserved; only the mask region is replaced.

    Returns:
        synthetic_frames: (N, 3, H, W) float32 [0,1]
    """
    logger.info("=" * 60)
    logger.info("[sub.06] ====== LTX Local Inpainting ======")
    logger.info(f"  frames:     {frames.shape}  dtype={frames.dtype}")
    logger.info(f"  masks:      {masks.shape}  dtype={masks.dtype}")
    logger.info(f"  control_pack: {control_pack.shape}  dtype={control_pack.dtype}")
    logger.info(f"  reference_dir: {reference_dir}")
    logger.info(f"  prompt:     {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    logger.info(f"  control_strength: {control_strength}")
    logger.info(f"  steps: {num_inference_steps}  seed: {seed}")

    # -- Validate inputs --
    N, C, H, W = frames.shape
    if C != 3:
        raise ValueError(f"frames must have 3 channels, got {C}")
    if masks.shape[0] != N or masks.shape[1] != 1 or masks.shape[2] != H or masks.shape[3] != W:
        raise ValueError(f"masks shape mismatch: {masks.shape} vs frames ({N},3,{H},{W})")
    if control_pack.shape[0] != N or control_pack.shape[1] != 4 or control_pack.shape[2] != H or control_pack.shape[3] != W:
        raise ValueError(f"control_pack shape mismatch: {control_pack.shape} vs frames ({N},3,{H},{W})")

    # -- Ensure float32 [0,1] --
    frames = _to_float32(frames)
    masks = _to_float32(masks)
    control_pack = _to_float32(control_pack)

    fps = int(metadata.get("fps", 30))
    orig_w = int(metadata.get("width", W))
    orig_h = int(metadata.get("height", H))

    # Align resolution to 64-multiple (LTX requirement)
    h_a = ((orig_h + 63) // 64) * 64
    w_a = ((orig_w + 63) // 64) * 64
    # Align frames to 8n+1
    remainder = (N - 1) % 8
    N_a = N if remainder == 0 else N + (8 - remainder)

    logger.info(f"[sub.06] Resolution: {orig_w}x{orig_h} -> aligned {w_a}x{h_a}")
    logger.info(f"[sub.06] Frames: {N} -> aligned {N_a} (8n+1)")

    # -- Build control mode string from metadata --
    control_mode = metadata.get("control_mode", "pose+depth+canny")

    # -- Generate control MP4 videos from control_pack channels --
    # Channel layout: R=Canny, G=Distance, B=Pose, A=Temporal
    # For IC-LoRA we need: pose (from B), depth (from G), canny (from R)
    control_dir = tempfile.mkdtemp(prefix="memento_controls_")
    control_videos = []
    try:
        # Pose heatmap (channel 2 = B)
        if "pose" in control_mode:
            pose_ch = control_pack[:, 2, :, :]  # (N, H, W)
            pose_mp4 = os.path.join(control_dir, "pose_control.mp4")
            _build_control_mp4_from_channel(pose_ch, pose_mp4, fps, "pose")
            control_videos.append((pose_mp4, control_strength))

        # Depth (channel 1 = G)
        if "depth" in control_mode:
            depth_ch = control_pack[:, 1, :, :]
            depth_mp4 = os.path.join(control_dir, "depth_control.mp4")
            _build_control_mp4_from_channel(depth_ch, depth_mp4, fps, "depth")
            control_videos.append((depth_mp4, control_strength))

        # Canny (channel 0 = R)
        if "canny" in control_mode:
            canny_ch = control_pack[:, 0, :, :]
            canny_mp4 = os.path.join(control_dir, "canny_control.mp4")
            _build_control_mp4_from_channel(canny_ch, canny_mp4, fps, "canny")
            control_videos.append((canny_mp4, control_strength))

        if not control_videos:
            raise RuntimeError(f"No control videos generated for mode={control_mode}")

        # -- Load reference images --
        if not os.path.isdir(reference_dir):
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

        from PIL import Image
        ref_files = sorted([
            f for f in os.listdir(reference_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        if not ref_files:
            raise RuntimeError(f"No reference images found in {reference_dir}")

        reference_images = []
        for fname in ref_files:
            img = Image.open(os.path.join(reference_dir, fname)).convert("RGB")
            reference_images.append(img)
        logger.info(f"[sub.06] Loaded {len(reference_images)} reference images")

        # -- Load pipeline --
        pipeline = _load_ltx_pipeline(control_mode, control_strength)

        # -- Prepare reference tensor --
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref_np = np.array(reference_images[0].resize((w_a, h_a), Image.LANCZOS)).astype(np.float32) / 255.0
        ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # -- Run inference --
        logger.info(f"[sub.06] Running LTX inference (seed={seed}, steps={num_inference_steps}, frames={N_a}, {w_a}x{h_a})...")
        infer_start = time.time()

        pipeline_kwargs = {
            "prompt": prompt,
            "seed": seed,
            "height": h_a,
            "width": w_a,
            "num_frames": N_a,
            "frame_rate": fps,
            "conditioning_attention_strength": control_strength,
            "video_conditioning": control_videos,
            "enhance_prompt": True,
            "skip_stage_2": False,
            "image": ref_tensor,
        }

        try:
            video_iterator, _ = pipeline(**pipeline_kwargs)
        except TypeError:
            logger.warning("[sub.06] Pipeline does not accept 'image' kwarg; retrying without it.")
            pipeline_kwargs.pop("image", None)
            video_iterator, _ = pipeline(**pipeline_kwargs)

        # -- Collect generated frames as tensors --
        gen_frames = []
        for i in range(min(N, N_a)):
            try:
                gen_frame = next(video_iterator)
            except StopIteration:
                logger.warning(f"[sub.06] Generator exhausted at frame {i + 1}/{N}")
                break
            if isinstance(gen_frame, torch.Tensor):
                gen_frame = gen_frame.detach()
                if gen_frame.device != device:
                    gen_frame = gen_frame.to(device)
            else:
                # numpy fallback
                gen_frame = torch.from_numpy(np.array(gen_frame).astype(np.float32) / 255.0).to(device)
            # Ensure CHW format
            if gen_frame.dim() == 4:
                gen_frame = gen_frame.squeeze(0)
            if gen_frame.shape[0] != 3:
                gen_frame = gen_frame.permute(2, 0, 1)
            # Resize to target resolution if needed
            if gen_frame.shape[1] != H or gen_frame.shape[2] != W:
                gen_frame = F.interpolate(
                    gen_frame.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0)
            # Ensure [0,1] range
            gen_frame = _to_float32(gen_frame)
            gen_frames.append(gen_frame)

        infer_elapsed = time.time() - infer_start
        logger.info(f"[sub.06] Inference complete: {len(gen_frames)} frames in {infer_elapsed:.1f}s")

        if not gen_frames:
            raise RuntimeError("[sub.06] No frames generated by LTX pipeline.")

        # -- Composite: generated_person * mask + original_frame * (1 - mask) --
        synthetic_frames = []
        for i in range(len(gen_frames)):
            gen = gen_frames[i]                     # (3, H, W)
            orig = frames[i]                        # (3, H, W)
            msk = masks[i]                          # (1, H, W)
            # Ensure mask has same spatial dims
            if msk.shape[1] != H or msk.shape[2] != W:
                msk = F.interpolate(msk.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            # Composite
            comp = gen * msk + orig * (1.0 - msk)
            synthetic_frames.append(comp)

        synthetic = torch.stack(synthetic_frames, dim=0)  # (N, 3, H, W)
        synthetic = _to_float32(synthetic)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.info(f"[sub.06] ====== LTX Inpainting done: {synthetic.shape} ======")
        return synthetic

    finally:
        shutil.rmtree(control_dir, ignore_errors=True)


# =============================================================================
# 07 -- RAFT Optical Flow Correction
# =============================================================================

def _load_raft_model() -> Tuple[object, torch.device]:
    """Load or return cached RAFT Large model."""
    cached_model = MODEL_CACHE.get("_raft_model")
    cached_device = MODEL_CACHE.get("_raft_device")
    if cached_model is not None and cached_device is not None:
        return cached_model, cached_device

    if not _RAFT_AVAILABLE:
        raise ImportError("RAFT model not available. Install torchvision >= 0.15.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    logger.info("[sub.07] Loading RAFT Large model...")

    raft_path = "/models/raft/raft_large.pth"
    if os.path.exists(raft_path):
        logger.info(f"[sub.07] Loading RAFT from local: {raft_path}")
        state_dict = torch.load(raft_path, map_location=device)
        model = raft_large(weights=None)
        model.load_state_dict(state_dict)
    else:
        logger.info("[sub.07] Loading RAFT from torchvision pretrained weights")
        model = raft_large(weights=Raft_Large_Weights.DEFAULT)

    model.to(device)
    model.eval()
    elapsed = time.time() - start
    logger.info(f"[sub.07] RAFT loaded in {elapsed:.1f}s on {device}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    MODEL_CACHE["_raft_model"] = model
    MODEL_CACHE["_raft_device"] = device
    return model, device


def _raft_compute_flow(
    model: object,
    img_a: torch.Tensor,  # (3, H, W) float32 [0,1]
    img_b: torch.Tensor,  # (3, H, W) float32 [0,1]
    device: torch.device,
) -> torch.Tensor:
    """Compute RAFT optical flow from img_a to img_b.

    Returns:
        flow: (2, H, W) float32, channels = (dx, dy) in pixel units
    """
    _, H, W = img_a.shape
    # Pad to multiple of 8
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8

    a_pad = F.pad(img_a.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")
    b_pad = F.pad(img_b.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect")
    a_pad = a_pad.to(device)
    b_pad = b_pad.to(device)

    with torch.no_grad():
        flow_list = model(a_pad, b_pad)
        flow = flow_list[-1]  # (1, 2, H_pad, W_pad)

    # Crop padding
    if pad_h > 0 or pad_w > 0:
        flow = flow[:, :, :H, :W]
    return flow.squeeze(0)


def _farneback_compute_flow(
    img_a: torch.Tensor,  # (3, H, W) float32 [0,1]
    img_b: torch.Tensor,  # (3, H, W) float32 [0,1]
) -> torch.Tensor:
    """Compute optical flow using Farneback (OpenCV fallback).

    Returns:
        flow: (2, H, W) float32 tensor on same device as input
    """
    if not _OPENCV_AVAILABLE:
        raise RuntimeError("OpenCV required for Farneback fallback.")

    device = img_a.device
    np_a = _tensor_to_numpy_frame(img_a)
    np_b = _tensor_to_numpy_frame(img_b)
    gray_a = cv2.cvtColor(np_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(np_b, cv2.COLOR_BGR2GRAY)

    flow_np = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )  # (H, W, 2)

    flow = torch.from_numpy(flow_np.astype(np.float32)).permute(2, 0, 1).to(device)
    return flow


def raft_correct(
    original_frames: torch.Tensor,   # (N, 3, H, W) float32 [0,1]
    synthetic_frames: torch.Tensor,  # (N, 3, H, W) float32 [0,1]
    masks: torch.Tensor,             # (N, 1, H, W) float32 [0,1]
    flow_diff_threshold: float = 2.0,
    mask_blend_strength: float = 0.85,
) -> torch.Tensor:
    """Node 07: RAFT optical-flow temporal correction.

    For each frame pair (t, t+1):
      1. Compute original_flow and synthetic_flow
      2. Compute flow difference magnitude
      3. Where difference > threshold, warp synthetic using original flow
      4. Mask-aware blending: blend_weight = mask * diff_mask * mask_blend_strength
    Then apply 3-frame temporal smoothing.

    Returns:
        aligned_frames: (N, 3, H, W) float32 [0,1]
    """
    logger.info("=" * 60)
    logger.info("[sub.07] ====== RAFT Optical Flow Correction ======")
    logger.info(f"  original:   {original_frames.shape}")
    logger.info(f"  synthetic:  {synthetic_frames.shape}")
    logger.info(f"  masks:      {masks.shape}")
    logger.info(f"  flow_diff_threshold: {flow_diff_threshold}")
    logger.info(f"  mask_blend_strength: {mask_blend_strength}")

    N, C, H, W = original_frames.shape
    if C != 3:
        raise ValueError(f"Expected 3-channel frames, got {C}")
    if synthetic_frames.shape != original_frames.shape:
        raise ValueError(f"synthetic_frames shape mismatch: {synthetic_frames.shape} vs {original_frames.shape}")
    if masks.shape[0] != N or masks.shape[1] != 1:
        raise ValueError(f"masks shape mismatch: {masks.shape}")

    # Ensure float32 [0,1]
    original_frames = _to_float32(original_frames)
    synthetic_frames = _to_float32(synthetic_frames)
    masks = _to_float32(masks)

    if N < 2:
        logger.warning("[sub.07] Fewer than 2 frames; returning synthetic as-is.")
        return synthetic_frames.clone()

    device = original_frames.device

    # -- Load RAFT model (with Farneback fallback) --
    use_raft = _RAFT_AVAILABLE
    raft_model = None
    raft_device = None
    if use_raft:
        try:
            raft_model, raft_device = _load_raft_model()
        except Exception as e:
            logger.warning(f"[sub.07] RAFT load failed: {e}; falling back to Farneback.")
            use_raft = False

    # -- Compute flows and corrections --
    corrected = []
    for i in range(N):
        if i == N - 1:
            # Last frame: use synthetic as-is
            corrected.append(synthetic_frames[i].clone())
            continue

        orig_a = original_frames[i]
        orig_b = original_frames[i + 1]
        synth_a = synthetic_frames[i]
        synth_b = synthetic_frames[i + 1]

        # Compute both flows
        if use_raft and raft_model is not None:
            try:
                orig_flow = _raft_compute_flow(raft_model, orig_a, orig_b, raft_device)
                synth_flow = _raft_compute_flow(raft_model, synth_a, synth_b, raft_device)
                # Move flows back to input device if needed
                orig_flow = orig_flow.to(device)
                synth_flow = synth_flow.to(device)
            except Exception as e:
                logger.warning(f"[sub.07] RAFT compute failed at frame {i}: {e}; fallback Farneback.")
                orig_flow = _farneback_compute_flow(orig_a, orig_b)
                synth_flow = _farneback_compute_flow(synth_a, synth_b)
        else:
            orig_flow = _farneback_compute_flow(orig_a, orig_b)
            synth_flow = _farneback_compute_flow(synth_a, synth_b)

        # Flow difference mask
        flow_diff = orig_flow - synth_flow
        diff_mag = _flow_magnitude(flow_diff.unsqueeze(0)).squeeze(0)  # (H, W)
        diff_mask = (diff_mag > flow_diff_threshold).float()  # (H, W)
        diff_mask = _apply_gaussian_blur(diff_mask.unsqueeze(0).unsqueeze(0), 5, 1.0).squeeze(0).squeeze(0)

        # Warp synthetic using original flow
        warped = _warp_tensor(synth_a, orig_flow)  # (3, H, W)

        # Mask-aware blending
        msk = masks[i].squeeze(0)  # (H, W)
        msk = _apply_gaussian_blur(msk.unsqueeze(0).unsqueeze(0), 7, 2.0).squeeze(0).squeeze(0)

        blend_w = msk * diff_mask * mask_blend_strength
        blend_w = blend_w.clamp(0.0, 1.0)

        # Blend: blend_w * warped + (1 - blend_w) * original
        corr = warped * blend_w + orig_a * (1.0 - blend_w)
        corrected.append(corr)

        if (i + 1) % 30 == 0 or i == N - 2:
            logger.info(f"[sub.07] Flow correction: {i + 1}/{N} frames")

    # -- Temporal smoothing (3-frame weighted average) --
    logger.info("[sub.07] Applying temporal smoothing...")
    temporal_weight = 0.33
    center_weight = 1.0 - 2.0 * temporal_weight
    aligned = []
    for i in range(N):
        prev_f = corrected[max(0, i - 1)]
        curr_f = corrected[i]
        next_f = corrected[min(N - 1, i + 1)]
        if i == 0:
            smoothed = curr_f * (1.0 - temporal_weight) + next_f * temporal_weight
        elif i == N - 1:
            smoothed = curr_f * (1.0 - temporal_weight) + prev_f * temporal_weight
        else:
            smoothed = prev_f * temporal_weight + curr_f * center_weight + next_f * temporal_weight
        aligned.append(smoothed)

    aligned_frames = torch.stack(aligned, dim=0)
    aligned_frames = _to_float32(aligned_frames)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"[sub.07] ====== RAFT correction done: {aligned_frames.shape} ======")
    return aligned_frames


# =============================================================================
# 08 -- Layered Fusion
# =============================================================================

def _histogram_match_torch(
    source: torch.Tensor,  # (3, H, W) float32 [0,1]
    target: torch.Tensor,  # (3, H, W) float32 [0,1]
) -> torch.Tensor:
    """Histogram-match source to target per-channel.  Works on GPU via numpy round-trip.

    Returns:
        matched: (3, H, W) float32 [0,1]
    """
    device = source.device
    src_np = (source.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    tgt_np = (target.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    result_np = np.zeros_like(src_np)
    for c in range(3):
        src_ch = src_np[c]
        tgt_ch = tgt_np[c]

        src_hist, _ = np.histogram(src_ch, 256, [0, 256])
        tgt_hist, _ = np.histogram(tgt_ch, 256, [0, 256])

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
        src_cdf = src_cdf / (src_cdf[-1] if src_cdf[-1] > 0 else 1)
        tgt_cdf = tgt_cdf / (tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1)

        lut = np.zeros(256, dtype=np.uint8)
        tj = 0
        for si in range(256):
            while tj < 256 and tgt_cdf[tj] < src_cdf[si]:
                tj += 1
            lut[si] = min(tj, 255)

        result_np[c] = lut[src_ch]

    result = torch.from_numpy(result_np.astype(np.float32) / 255.0).to(device)
    return result.clamp(0.0, 1.0)


def _laplacian_pyramid_blend(
    fg: torch.Tensor,   # (3, H, W) float32 [0,1]
    bg: torch.Tensor,   # (3, H, W) float32 [0,1]
    mask: torch.Tensor, # (H, W) float32 [0,1]
    depth: int = 4,
) -> torch.Tensor:
    """Laplacian pyramid edge-preserving blend.

    Uses numpy/OpenCV round-trip for the pyramid construction, which is
    more reliable than pure-PyTorch for this image-processing task.
    """
    if not _OPENCV_AVAILABLE:
        # Fallback to simple alpha blend
        m3 = mask.unsqueeze(0).clamp(0, 1)
        return fg * m3 + bg * (1.0 - m3)

    device = fg.device
    np_fg = _tensor_to_numpy_frame(fg)
    np_bg = _tensor_to_numpy_frame(bg)
    np_mask = _tensor_to_numpy_gray(mask).astype(np.float32) / 255.0

    # Build pyramids
    gauss_fg = [np_fg.astype(np.float32)]
    gauss_bg = [np_bg.astype(np.float32)]
    gauss_m = [np_mask]

    for i in range(depth):
        gauss_fg.append(cv2.pyrDown(gauss_fg[-1]))
        gauss_bg.append(cv2.pyrDown(gauss_bg[-1]))
        gauss_m.append(cv2.pyrDown(gauss_m[-1]))

    laplace_fg = []
    laplace_bg = []
    for i in range(depth):
        up = cv2.pyrUp(gauss_fg[i + 1])
        hh, ww = gauss_fg[i].shape[:2]
        up = cv2.resize(up, (ww, hh))
        laplace_fg.append(gauss_fg[i] - up)

        up = cv2.pyrUp(gauss_bg[i + 1])
        up = cv2.resize(up, (ww, hh))
        laplace_bg.append(gauss_bg[i] - up)

    laplace_fg.append(gauss_fg[-1])
    laplace_bg.append(gauss_bg[-1])

    # Composite each level
    composite = []
    for i in range(depth + 1):
        hh, ww = laplace_fg[i].shape[:2]
        gm = cv2.resize(gauss_m[min(i, depth)], (ww, hh))
        gm3 = np.dstack([gm, gm, gm]) if gm.ndim == 2 else gm
        lvl = laplace_fg[i] * gm3 + laplace_bg[i] * (1.0 - gm3)
        composite.append(lvl)

    # Reconstruct
    result = composite[-1]
    for i in range(depth - 1, -1, -1):
        result = cv2.pyrUp(result)
        hh, ww = composite[i].shape[:2]
        result = cv2.resize(result, (ww, hh))
        result += composite[i]

    result = np.clip(result, 0, 255).astype(np.uint8)
    return _numpy_frame_to_tensor(result, device)


def _screen_blend(fg: torch.Tensor, bg: torch.Tensor) -> torch.Tensor:
    """Screen blend mode: 1 - (1 - fg) * (1 - bg)."""
    return 1.0 - (1.0 - fg) * (1.0 - bg)


def fusion_blend(
    synthetic_frames: torch.Tensor,  # (N, 3, H, W) float32 [0,1] from node 07
    masks: torch.Tensor,             # (N, 1, H, W) float32 [0,1] from node 02
    depth_maps: torch.Tensor,        # (N, 1, H, W) float32 [0,1] from node 04
) -> torch.Tensor:
    """Node 08: 4-layer lighting / colour fusion.

    NOTE: Does NOT need original frames -- node 07 output already preserves
    background.

    4-layer blending:
      Layer 0 (foreground): direct replacement where mask > 0.5
      Layer 1 (feather):    Gaussian blur on mask edges, alpha blend
      Layer 2 (detail):     Laplacian pyramid edge-preserving blend
      Layer 3 (semitrans):  screen blend mode

    Colour matching: histogram matching of foreground to background boundary.
    Depth-aware shadow: deeper depth -> darker shadow (multiply 0.5~1.0).

    Returns:
        final_frames: (N, 3, H, W) float32 [0,1]
    """
    logger.info("=" * 60)
    logger.info("[sub.08] ====== Layered Fusion ======")
    logger.info(f"  synthetic: {synthetic_frames.shape}")
    logger.info(f"  masks:     {masks.shape}")
    logger.info(f"  depth:     {depth_maps.shape}")

    N, C, H, W = synthetic_frames.shape
    if C != 3:
        raise ValueError(f"Expected 3-channel frames, got {C}")

    synthetic_frames = _to_float32(synthetic_frames)
    masks = _to_float32(masks)
    if depth_maps is not None:
        depth_maps = _to_float32(depth_maps)

    device = synthetic_frames.device
    final_frames_list = []

    for i in range(N):
        fg = synthetic_frames[i]          # (3, H, W)
        msk = masks[i].squeeze(0)         # (H, W)
        depth = depth_maps[i].squeeze(0) if depth_maps is not None else None  # (H, W)

        # -- Since node 07 preserves background, fg is already the composited frame.
        #    We treat fg as both foreground and background for the purpose of
        #    edge refinement.  The "background" is the surrounding region of fg
        #    outside the mask.
        #    Composite = fg  (already correct from node 07)

        # -- Layer 0: Foreground direct replacement --
        # The mask > 0.5 region is the character region that should be kept as-is.
        layer0_mask = (msk > 0.5).float()  # (H, W)
        # For this node, since fg already has the correct background, we blend
        # the layers to refine edges.  We start with fg as the base.
        composite = fg.clone()

        # -- Colour matching: match fg (mask region) to boundary region --
        # Dilate and erode to get boundary
        if _OPENCV_AVAILABLE:
            np_mask = _tensor_to_numpy_gray(msk)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            np_dilated = cv2.dilate(np_mask, kernel, iterations=3)
            np_eroded = cv2.erode(np_mask, kernel, iterations=3)
            boundary = np.clip(np_dilated.astype(np.float32) - np_eroded.astype(np.float32), 0, 255).astype(np.uint8)
            boundary_t = _numpy_gray_to_tensor(boundary, device).squeeze(0)  # (H, W)

            if boundary_t.sum() > 10:
                # Build background boundary sample
                bg_boundary = fg * boundary_t.clamp(0, 1)
                fg_matched = _histogram_match_torch(fg, bg_boundary)
            else:
                fg_matched = fg
        else:
            fg_matched = fg

        # -- Depth-aware shadow --
        if depth is not None:
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = depth
            shadow = 1.0 - 0.5 * depth_norm  # range [0.5, 1.0]
            # Apply shadow only in mask region
            fg_matched = fg_matched * (shadow * msk + (1.0 - msk))
            fg_matched = fg_matched.clamp(0.0, 1.0)

        # -- Layer 1: Feather (Gaussian blur on mask edges) --
        feather_mask = _apply_gaussian_blur(msk.unsqueeze(0).unsqueeze(0), 21, 7.0).squeeze(0).squeeze(0)
        # Only apply feather in the edge region (where mask is between 0.1 and 0.9)
        feather_region = ((msk > 0.1) & (msk < 0.9)).float()
        feather_mask = feather_mask * feather_region
        composite = fg_matched * feather_mask + composite * (1.0 - feather_mask)

        # -- Layer 2: Detail (Laplacian pyramid edge-preserving) --
        detail_region = ((msk > 0.05) & (msk < 0.95)).float()
        if detail_region.sum() > 100:
            composite = _laplacian_pyramid_blend(fg_matched, composite, detail_region)

        # -- Layer 3: Semitransparent (screen blend) --
        semi_mask = _apply_gaussian_blur(msk.unsqueeze(0).unsqueeze(0), 11, 4.0).squeeze(0).squeeze(0)
        semi_region = ((msk > 0.0) & (msk < 0.3)).float()
        semi_mask = semi_mask * semi_region
        screen_result = _screen_blend(fg_matched, composite)
        composite = fg_matched * semi_mask + screen_result * (1.0 - semi_mask)

        composite = composite.clamp(0.0, 1.0)
        final_frames_list.append(composite)

        if (i + 1) % max(1, N // 10) == 0 or i == N - 1:
            logger.info(f"[sub.08] Fusion progress: {i + 1}/{N} frames")

    final_frames = torch.stack(final_frames_list, dim=0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"[sub.08] ====== Fusion done: {final_frames.shape} ======")
    return final_frames


# =============================================================================
# 09 -- FFmpeg Composite
# =============================================================================

def composite_video(
    frames: torch.Tensor,          # (N, 3, H, W) float32 [0,1]
    audio_path: str,               # path to audio file
    metadata: dict,                # {"fps": 30, "width": 1920, "height": 1080}
    output_path: str = "/workspace/output.mp4",
) -> str:
    """Node 09: FFmpeg final composite.

    Saves frames as PNG sequence to a temp directory, then runs FFmpeg to
    combine frames + audio into a video.  Supports up to 4K resolution.

    Returns:
        output_path: path to the generated MP4 file
    """
    logger.info("=" * 60)
    logger.info("[sub.09] ====== FFmpeg Composite ======")
    logger.info(f"  frames:     {frames.shape}")
    logger.info(f"  audio_path: {audio_path}")
    logger.info(f"  metadata:   {metadata}")
    logger.info(f"  output:     {output_path}")

    if not _OPENCV_AVAILABLE:
        raise RuntimeError("[sub.09] OpenCV is required for frame encoding.")

    N, C, H, W = frames.shape
    if C != 3:
        raise ValueError(f"Expected 3-channel frames, got {C}")

    frames = _to_float32(frames)

    fps = float(metadata.get("fps", 30))
    vid_width = int(metadata.get("width", W))
    vid_height = int(metadata.get("height", H))

    # -- Determine output directory --
    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    # -- Write frames to temp directory --
    tmp_dir = tempfile.mkdtemp(prefix="memento_composite_")
    try:
        logger.info(f"[sub.09] Writing {N} frames to temp dir: {tmp_dir}")
        for i in range(N):
            frame_np = _tensor_to_numpy_frame(frames[i])
            # Resize to target resolution if needed
            if frame_np.shape[0] != vid_height or frame_np.shape[1] != vid_width:
                frame_np = cv2.resize(frame_np, (vid_width, vid_height), interpolation=cv2.INTER_LANCZOS4)
            out_frame_path = os.path.join(tmp_dir, f"frame_{i + 1:06d}.png")
            cv2.imwrite(out_frame_path, frame_np)

        # -- Build FFmpeg command --
        ffmpeg = "ffmpeg"
        # Verify ffmpeg exists
        try:
            subprocess.run([ffmpeg, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            # Try common paths
            for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
                try:
                    subprocess.run([candidate, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    ffmpeg = candidate
                    break
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            else:
                raise RuntimeError("FFmpeg not found. Please install FFmpeg.")

        has_audio = audio_path and os.path.isfile(audio_path)
        if has_audio:
            # Verify audio stream
            try:
                probe = subprocess.run(
                    [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error",
                     "-select_streams", "a:0", "-show_entries", "stream=codec_type",
                     "-of", "csv=p=0", audio_path],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
                )
                has_audio = "audio" in probe.stdout.lower()
            except Exception:
                has_audio = os.path.getsize(audio_path) > 0

        cmd = [
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%06d.png"),
        ]

        if has_audio:
            logger.info(f"[sub.09] Audio source: {audio_path}")
            cmd += ["-i", audio_path]
            cmd += [
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "320k",
                "-shortest",
            ]
        else:
            logger.info("[sub.09] No audio; producing silent video.")
            cmd += [
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "slow",
                "-pix_fmt", "yuv420p",
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
            ]

        # Scale filter for 4K support
        vf_parts = [
            f"scale={vid_width}:{vid_height}:force_original_aspect_ratio=decrease",
            f"pad={vid_width}:{vid_height}:(ow-iw)/2:(oh-ih)/2",
        ]
        cmd.insert(-1 if has_audio else -1, "-vf")
        cmd.insert(-1 if has_audio else -1, ",".join(vf_parts))

        cmd.append(output_path)

        logger.info(f"[sub.09] FFmpeg: {' '.join(cmd)}")

        # -- Run FFmpeg --
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (code {result.returncode}): {result.stderr[-2000:]}"
            )

        # -- Verify output --
        if not os.path.isfile(output_path):
            raise RuntimeError(f"Output file not created: {output_path}")

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"[sub.09] Output: {output_path}  ({file_size_mb:.1f} MB)")

        # Probe duration
        try:
            probe = subprocess.run(
                [ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error",
                 "-show_entries", "format=duration", "-of", "csv=p=0", output_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                dur = float(probe.stdout.strip())
                mins, secs = divmod(dur, 60)
                hours, mins = divmod(mins, 60)
                logger.info(f"[sub.09] Duration: {int(hours)}h {int(mins)}m {secs:.1f}s")
        except Exception:
            pass

        logger.info(f"[sub.09] ====== Composite done: {output_path} ======")
        return output_path

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)