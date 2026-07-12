"""
GPU Tensor-Based Operations for Memento Pipeline — Nodes 02–05.

All functions accept torch.Tensor and return torch.Tensor (or torch.Tensor + dict).
No file I/O except model loading.  Module-level singleton caches for all models.

Nodes:
  02 — SAM2.1 Video Segmentation
  03 — MediaPipe 2D Pose Detection + Heatmap
  04 — MotionBERT 2D→3D Lifting + Depth Map
  05 — Align Control Signals (4-channel control pack)
"""

from __future__ import annotations

import logging
import math
import pickle
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache  (singletons shared across calls)
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# 4-layer SVG mask data storage  (populated by segment_video, consumed later)
# ---------------------------------------------------------------------------
_SVG_MASK_DATA: Dict[str, Any] = {}


def _get_device(device: str) -> torch.device:
    """Resolve device string, falling back to CUDA if available."""
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _ensure_float32(t: torch.Tensor) -> torch.Tensor:
    """Cast tensor to float32 and clamp to [0, 1]."""
    return t.to(torch.float32).clamp(0.0, 1.0)


def _preprocess_frames(frames: torch.Tensor) -> torch.Tensor:
    """
    Ensure frames are (N, 3, H, W) float32 [0, 1].
    If input is uint8 [0, 255], convert to float32 [0, 1].
    If input is (N, H, W, C), permute to (N, C, H, W).
    """
    if frames.dtype == torch.uint8:
        frames = frames.to(torch.float32) / 255.0
    else:
        frames = frames.to(torch.float32)
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    return frames


# ===================================================================
# Internal helpers
# ===================================================================


def _normalize_coords(
    coords: List[Tuple[float, float, int]], H: int, W: int
) -> List[Tuple[float, float, int]]:
    """Normalize click coordinates from pixel space to [0, 1]."""
    return [(x / W, y / H, label) for (x, y, label) in coords]


def _draw_gaussian_spot(
    canvas: np.ndarray, cx: float, cy: float, sigma: float = 6.0, radius: int = 25
) -> None:
    """Draw a Gaussian spot on a numpy canvas in-place."""
    H, W = canvas.shape
    x0 = max(0, int(cx - radius))
    x1 = min(W, int(cx + radius + 1))
    y0 = max(0, int(cy - radius))
    y1 = min(H, int(cy + radius + 1))
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    g = np.exp(-0.5 * dist2 / (sigma * sigma))
    canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)


def _draw_line_gaussian(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    sigma: float = 4.0,
    num_samples: int = 100,
) -> None:
    """Draw a Gaussian-blurred line between two points on a numpy canvas."""
    for t in np.linspace(0, 1, num_samples):
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        _draw_gaussian_spot(canvas, cx, cy, sigma=sigma, radius=18)


# ---------------------------------------------------------------------------
# MediaPipe skeleton connectivity (33 keypoints)
# ---------------------------------------------------------------------------
_MEDIAPIPE_SKELETON: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7),         # head ↔ left arm
    (0, 4), (4, 5), (5, 6), (6, 8),          # head ↔ right arm
    (9, 10),                                  # mouth
    (11, 12),                                 # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # right arm
    (11, 23), (12, 24), (23, 24),             # torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # right leg
]

# MediaPipe 33 → H36M 17 mapping
# H36M 17: 0-hip, 1-rhip, 2-rknee, 3-rankle, 4-lhip, 5-lknee, 6-lankle,
#          7-spine, 8-thorax, 9-neck, 10-head, 11-lshoulder,12-lelbow,13-lwrist,
#          14-rshoulder,15-relbow,16-rwrist
_MEDIAPIPE_TO_H36M: Dict[int, int] = {
    0: 10,    # nose → head
    11: 11,   # left shoulder
    12: 14,   # right shoulder
    13: 12,   # left elbow
    14: 15,   # right elbow
    15: 13,   # left wrist
    16: 16,   # right wrist
    23: 4,    # left hip
    24: 1,    # right hip
    25: 5,    # left knee
    26: 2,    # right knee
    27: 6,    # left ankle
    28: 3,    # right ankle
    29: 6,    # left heel → left ankle (approx)
    30: 3,    # right heel → right ankle (approx)
    31: 6,    # left foot index → left ankle
    32: 3,    # right foot index → right ankle
    # spine (7) and thorax (8) are approximated from hips & shoulders
}


# ===================================================================
# SimplePoseLifter — lightweight FC network for 2D → 3D lifting
# ===================================================================


class SimplePoseLifter(nn.Module):
    """Lightweight fully-connected network that lifts 2D keypoints to 3D."""

    def __init__(self, num_joints: int = 17, hidden_dim: int = 1024, num_blocks: int = 2):
        super().__init__()
        self.num_joints = num_joints
        input_dim = num_joints * 2  # x, y

        layers: List[nn.Module] = []
        in_dim = input_dim
        for i in range(num_blocks):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.25))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(hidden_dim, num_joints * 3)  # x, y, z

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, kp_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kp_2d: (N, num_joints * 2)  — flattened 2D keypoints
        Returns:
            kp_3d: (N, num_joints, 3)   — 3D keypoints
        """
        feat = self.encoder(kp_2d)
        out = self.decoder(feat)
        return out.view(-1, self.num_joints, 3)


# ===================================================================
# 02 — SAM2 Video Segmentation
# ===================================================================


def segment_video(
    frames: torch.Tensor,
    click_points: List[Tuple[float, float, int]],
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run SAM2.1 video segmentation on a sequence of frames.

    SAM2.1 Hiera-Large — Apache 2.0 license, fully open, no auth required.

    Args:
        frames:      (N, 3, H, W) float32 [0, 1].
        click_points: List of (x, y, label) in pixel coordinates.
        device:       Torch device string.

    Returns:
        masks: (N, 1, H, W) float32 [0, 1] binary segmentation masks.
    """
    logger.info("SAM2 segment_video: %d frames, %d click points", frames.shape[0], len(click_points))

    frames = _preprocess_frames(frames)
    N, C, H, W = frames.shape
    dev = _get_device(device)

    if N == 0:
        logger.warning("segment_video: received empty frames tensor.")
        return torch.zeros(0, 1, 1, 1, dtype=torch.float32)

    predictor = _get_sam2_predictor(dev)

    try:
        masks = _run_sam2_inference(predictor, frames, click_points, H, W, dev)
    except Exception as exc:
        logger.error("SAM2 inference failed: %s", exc)
        masks = torch.ones(N, 1, H, W, dtype=torch.float32, device=dev)
        logger.warning("Using fallback all-ones masks.")

    _generate_svg_mask_layers(masks)

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    logger.info("SAM2 segment_video: output masks shape %s", masks.shape)
    return masks


def _get_sam2_predictor(device: torch.device) -> Any:
    """Load and cache SAM2.1 video predictor singleton."""
    global _MODEL_CACHE, _CACHE_LOCK

    cache_key = f"sam2_video:{device}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        try:
            from sam2.build_sam import build_sam2_video_predictor

            checkpoint = "/models/sam2/sam2.1_hiera_large.pt"
            model_cfg = "sam2.1_hiera_l.yaml"
            logger.info("Loading SAM2.1 from %s ...", checkpoint)

            predictor = build_sam2_video_predictor(
                model_cfg, checkpoint, device=str(device),
            )

            _MODEL_CACHE[cache_key] = predictor
            logger.info("SAM2.1 model loaded and cached successfully.")
            return predictor

        except FileNotFoundError:
            logger.error("SAM2 checkpoint not found at %s. Using dummy predictor.", checkpoint)
            predictor = _DummySAM2Predictor(device)
            _MODEL_CACHE[cache_key] = predictor
            return predictor
        except ImportError as exc:
            logger.error("sam2 package not installed: %s. Using dummy predictor.", exc)
            predictor = _DummySAM2Predictor(device)
            _MODEL_CACHE[cache_key] = predictor
            return predictor


class _DummySAM2Predictor:
    """Fallback predictor when SAM2 is unavailable; returns all-ones masks."""

    def __init__(self, device: torch.device):
        self.device = device

    def init_state(self, video_path: str) -> None:
        pass

    def add_new_points_or_box(self, *args, **kwargs):
        return (0, [1], torch.ones(1, 1, 256, 256, device=self.device))

    def add_new_points(self, *args, **kwargs):
        return (None, [1], torch.ones(1, 1, 256, 256, device=self.device), torch.ones(1, 1, 256, 256, device=self.device))

    def propagate_in_video(self, *args, **kwargs):
        yield 0, [1], torch.ones(1, 1, 256, 256, device=self.device)
        yield 1, [1], torch.ones(1, 1, 256, 256, device=self.device)

    def reset_state(self, *args, **kwargs) -> None:
        pass


def _run_sam2_inference(
    predictor: Any,
    frames: torch.Tensor,
    click_points: List[Tuple[float, float, int]],
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Execute SAM2.1 video inference on the frame sequence."""
    N = frames.shape[0]
    is_dummy = isinstance(predictor, _DummySAM2Predictor)

    if is_dummy:
        return torch.ones(N, 1, H, W, dtype=torch.float32, device=device)

    import tempfile, os, cv2
    tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        for i in range(N):
            img = (frames[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(tmpdir, f"{i:05d}.jpg"), img)

        with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmpdir)

            for x, y, label in click_points:
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=1,
                    points=[[x, y]], labels=[label],
                )

            masks = torch.zeros(N, 1, H, W, dtype=torch.float32, device=device)
            for frame_idx, obj_ids, out_masks in predictor.propagate_in_video(state):
                if 1 in obj_ids and out_masks is not None:
                    idx = obj_ids.index(1)
                    m = out_masks[idx]
                    if m.ndim == 2:
                        m = m.unsqueeze(0)
                    m = m[:1].float()
                    if m.shape[-2:] != (H, W):
                        m = F.interpolate(m.unsqueeze(0), size=(H, W), mode="bilinear").squeeze(0)
                    masks[frame_idx] = m.clamp(0, 1)

            predictor.reset_state(state)

    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return masks

# ===================================================================
# 03 — MediaPipe 2D Pose Detection + Heatmap
# ===================================================================


def extract_pose_2d(
    frames: torch.Tensor,
    masks: torch.Tensor,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Detect 2D pose landmarks using MediaPipe and generate pose heatmaps.

    Args:
        frames:   (N, 3, H, W) float32 [0, 1].
        masks:    (N, 1, H, W) float32 [0, 1] — regions to process.
        min_detection_confidence:  MediaPipe detection threshold.
        min_tracking_confidence:   MediaPipe tracking threshold.

    Returns:
        keypoints_dict:  {f"frame_{i+1:05d}": {"x": [...33], "y": [...33], "z": [...33], "visibility": [...33]}}
        heatmaps_tensor: (N, 1, H, W) float32 [0, 1] — pose heatmaps within mask region.
    """
    logger.info(
        "extract_pose_2d: %d frames, detection_conf=%.2f, tracking_conf=%.2f",
        frames.shape[0],
        min_detection_confidence,
        min_tracking_confidence,
    )

    frames = _preprocess_frames(frames)
    N, C, H, W = frames.shape
    device = frames.device

    if N == 0:
        return {}, torch.zeros(0, 1, 1, 1, dtype=torch.float32, device=device)

    # Ensure masks match
    if masks.shape[0] != N or masks.shape[-2:] != (H, W):
        masks = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        if masks.shape[0] == 1:
            masks = masks.expand(N, -1, -1, -1)

    # Convert to numpy for MediaPipe
    frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)  # (N, 3, H, W)
    frames_np = frames_np.transpose(0, 2, 3, 1)  # → (N, H, W, 3)
    masks_np = masks.cpu().squeeze(1).numpy()  # (N, H, W)

    keypoints_dict: Dict[str, Any] = {}

    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        Pose = mp_pose.Pose

        pose = Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
    except ImportError:
        logger.error("mediapipe not installed. Returning empty keypoints and zero heatmaps.")
        empty_heatmaps = torch.zeros(N, 1, H, W, dtype=torch.float32, device=device)
        return _empty_keypoints_dict(N), empty_heatmaps

    all_keypoints: List[Optional[np.ndarray]] = []  # each: (33, 4) or None

    for i in range(N):
        img = frames_np[i].copy()
        mask = masks_np[i]

        # Apply mask: set non-mask regions to black to reduce interference
        if mask.ndim == 2:
            mask_3ch = np.stack([mask] * 3, axis=-1)
            img = (img.astype(np.float32) * mask_3ch).astype(np.uint8)

        # Ensure RGB
        if img.shape[-1] == 3:
            img_rgb = img
        else:
            img_rgb = img

        try:
            results = pose.process(img_rgb)
        except Exception as e:
            logger.warning("MediaPipe processing failed on frame %d: %s", i, e)
            results = None

        if results is not None and results.pose_landmarks is not None:
            lm = results.pose_landmarks
            kp = np.zeros((33, 4), dtype=np.float32)
            for j, landmark in enumerate(lm.landmark):
                kp[j, 0] = landmark.x * W
                kp[j, 1] = landmark.y * H
                kp[j, 2] = landmark.z
                kp[j, 3] = landmark.visibility
            all_keypoints.append(kp)

            keypoints_dict[f"frame_{i+1:05d}"] = {
                "x": kp[:, 0].tolist(),
                "y": kp[:, 1].tolist(),
                "z": kp[:, 2].tolist(),
                "visibility": kp[:, 3].tolist(),
            }
        else:
            all_keypoints.append(None)
            keypoints_dict[f"frame_{i+1:05d}"] = {
                "x": [0.0] * 33,
                "y": [0.0] * 33,
                "z": [0.0] * 33,
                "visibility": [0.0] * 33,
            }

    pose.close()

    # Generate heatmaps
    heatmaps_tensor = _generate_pose_heatmaps(all_keypoints, masks_np, H, W, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logger.info("extract_pose_2d: %d frames processed, output heatmaps shape %s", N, heatmaps_tensor.shape)
    return keypoints_dict, heatmaps_tensor


def _empty_keypoints_dict(N: int) -> Dict[str, Any]:
    """Return a keypoints dict with all-zero entries for N frames."""
    d: Dict[str, Any] = {}
    for i in range(N):
        d[f"frame_{i+1:05d}"] = {
            "x": [0.0] * 33,
            "y": [0.0] * 33,
            "z": [0.0] * 33,
            "visibility": [0.0] * 33,
        }
    return d


def _generate_pose_heatmaps(
    all_keypoints: List[Optional[np.ndarray]],
    masks_np: np.ndarray,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate pose heatmaps from 2D keypoints with gaussian spots + skeleton lines."""
    N = len(all_keypoints)
    heatmaps = np.zeros((N, H, W), dtype=np.float32)

    for i in range(N):
        canvas = np.zeros((H, W), dtype=np.float32)
        kp = all_keypoints[i]
        if kp is None:
            heatmaps[i] = canvas
            continue

        # Draw keypoints
        for j in range(kp.shape[0]):
            v = kp[j, 3]
            if v < 0.3:
                continue
            cx, cy = kp[j, 0], kp[j, 1]
            if 0 <= cx < W and 0 <= cy < H:
                sigma = 4.0 if j < 11 else 5.0  # face joints smaller, body larger
                _draw_gaussian_spot(canvas, cx, cy, sigma=sigma, radius=20)

        # Draw skeleton lines
        for (a, b) in _MEDIAPIPE_SKELETON:
            va = kp[a, 3] if a < kp.shape[0] else 0.0
            vb = kp[b, 3] if b < kp.shape[0] else 0.0
            if va < 0.3 or vb < 0.3:
                continue
            x0, y0 = kp[a, 0], kp[a, 1]
            x1, y1 = kp[b, 0], kp[b, 1]
            if not (0 <= x0 < W and 0 <= y0 < H and 0 <= x1 < W and 0 <= y1 < H):
                continue
            _draw_line_gaussian(canvas, x0, y0, x1, y1, sigma=3.5)

        # Mask to mask region
        if i < len(masks_np):
            mask = masks_np[i]
            canvas = canvas * mask

        heatmaps[i] = canvas

    heatmaps_tensor = torch.from_numpy(heatmaps).unsqueeze(1).to(device)
    heatmaps_tensor = _ensure_float32(heatmaps_tensor)
    return heatmaps_tensor


# ===================================================================
# 04 — MotionBERT 2D→3D Lifting + Depth Map
# ===================================================================


def lift_pose_3d(
    keypoints_dict: Dict[str, Any],
    masks: torch.Tensor,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Lift 2D MediaPipe keypoints to 3D using SimplePoseLifter and generate depth maps.

    Args:
        keypoints_dict:  Output from extract_pose_2d.
        masks:           (N, 1, H, W) float32 [0, 1].

    Returns:
        pose3d_dict:  {f"frame_{i+1:05d}": {"x": [...17], "y": [...17], "z": [...17]}}
        depth_tensor: (N, 1, H, W) float32 [0, 1] — depth maps.
    """
    logger.info("lift_pose_3d: processing %d frames", len(keypoints_dict))

    N = len(keypoints_dict)
    if N == 0 or masks.shape[0] == 0:
        return {}, torch.zeros(0, 1, 1, 1, dtype=torch.float32, device=masks.device)

    device = masks.device
    _, _, H, W = masks.shape

    # Build 2D keypoints tensor (N, 17, 2)
    kp_2d_list: List[np.ndarray] = []
    for i in range(N):
        frame_key = f"frame_{i+1:05d}"
        if frame_key in keypoints_dict:
            kp = keypoints_dict[frame_key]
            kp_33 = np.stack([kp["x"], kp["y"], kp["visibility"]], axis=-1)  # (33, 3)
        else:
            kp_33 = np.zeros((33, 3), dtype=np.float32)
        kp_17 = _convert_mediapipe_to_h36m(kp_33)
        kp_2d_list.append(kp_17[:, :2])  # (17, 2)

    kp_2d = np.stack(kp_2d_list, axis=0)  # (N, 17, 2)
    kp_2d_t = torch.from_numpy(kp_2d).to(device)

    # Load / get SimplePoseLifter
    lifter = _get_pose_lifter(device)

    # Forward pass
    with torch.no_grad():
        lifter.eval()
        kp_2d_flat = kp_2d_t.view(N, -1)  # (N, 34)
        kp_3d = lifter(kp_2d_flat)  # (N, 17, 3)

    # Height normalization
    kp_3d = _height_normalize(kp_3d)

    kp_3d_np = kp_3d.cpu().numpy()

    pose3d_dict: Dict[str, Any] = {}
    for i in range(N):
        frame_key = f"frame_{i+1:05d}"
        pose3d_dict[frame_key] = {
            "x": kp_3d_np[i, :, 0].tolist(),
            "y": kp_3d_np[i, :, 1].tolist(),
            "z": kp_3d_np[i, :, 2].tolist(),
        }

    # Generate depth maps
    depth_tensor = _generate_depth_maps(masks, kp_3d_np, kp_2d, device)

    logger.info("lift_pose_3d: output pose3d_dict %d entries, depth shape %s", len(pose3d_dict), depth_tensor.shape)
    return pose3d_dict, depth_tensor


def _convert_mediapipe_to_h36m(kp_33: np.ndarray) -> np.ndarray:
    """
    Convert MediaPipe 33 keypoints (x, y, visibility) to H36M 17 keypoints (x, y, z).
    For missing H36M joints, we approximate.
    """
    kp_17 = np.zeros((17, 3), dtype=np.float32)

    for mp_idx, h36m_idx in _MEDIAPIPE_TO_H36M.items():
        if mp_idx < len(kp_33):
            kp_17[h36m_idx, :2] = kp_33[mp_idx, :2]
            kp_17[h36m_idx, 2] = kp_33[mp_idx, 2]  # visibility as z placeholder

    # Spine (7): midpoint of hips
    if kp_33[23, 2] > 0.01 and kp_33[24, 2] > 0.01:
        kp_17[7, :2] = (kp_33[23, :2] + kp_33[24, :2]) / 2.0
        kp_17[7, 2] = (kp_33[23, 2] + kp_33[24, 2]) / 2.0

    # Thorax (8): midpoint of shoulders
    if kp_33[11, 2] > 0.01 and kp_33[12, 2] > 0.01:
        kp_17[8, :2] = (kp_33[11, :2] + kp_33[12, :2]) / 2.0
        kp_17[8, 2] = (kp_33[11, 2] + kp_33[12, 2]) / 2.0

    # Neck (9): midpoint of head and thorax
    kp_17[9, :2] = (kp_17[10, :2] + kp_17[8, :2]) / 2.0
    kp_17[9, 2] = (kp_17[10, 2] + kp_17[8, 2]) / 2.0

    return kp_17


def _get_pose_lifter(device: torch.device) -> SimplePoseLifter:
    """Load and cache SimplePoseLifter singleton."""
    global _MODEL_CACHE, _CACHE_LOCK

    cache_key = "simple_pose_lifter"
    if cache_key in _MODEL_CACHE:
        lifter = _MODEL_CACHE[cache_key]
        if next(lifter.parameters()).device != device:
            lifter = lifter.to(device)
            _MODEL_CACHE[cache_key] = lifter
        return lifter

    with _CACHE_LOCK:
        if cache_key in _MODEL_CACHE:
            lifter = _MODEL_CACHE[cache_key]
            if next(lifter.parameters()).device != device:
                lifter = lifter.to(device)
                _MODEL_CACHE[cache_key] = lifter
            return lifter

        lifter = SimplePoseLifter(num_joints=17, hidden_dim=1024, num_blocks=2)
        lifter.to(device)
        lifter.eval()

        # Try to load pretrained weights
        weight_path = "/models/motionbert/pose_lifter.pth"
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
            lifter.load_state_dict(state, strict=False)
            logger.info("Loaded SimplePoseLifter weights from %s", weight_path)
        except FileNotFoundError:
            logger.warning(
                "Pretrained weights not found at %s. Using randomly initialized SimplePoseLifter.",
                weight_path,
            )
        except Exception as exc:
            logger.warning("Failed to load pose lifter weights: %s. Using random init.", exc)

        _MODEL_CACHE[cache_key] = lifter
        return lifter


def _height_normalize(kp_3d: torch.Tensor) -> torch.Tensor:
    """
    Normalize 3D pose by hip height so that the distance between
    hip (index 0) and head (index 10) is approximately 1.0.
    """
    hip = kp_3d[:, 0:1, :]  # (N, 1, 3)
    head = kp_3d[:, 10:11, :]
    height = torch.norm(head - hip, dim=-1, keepdim=True)  # (N, 1, 1)
    height = torch.clamp(height, min=1e-6)
    kp_3d = kp_3d / height
    return kp_3d


def _generate_depth_maps(
    masks: torch.Tensor,
    kp_3d_np: np.ndarray,
    kp_2d_np: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate depth maps by combining distance transform and RBF interpolation of 3D poses.

    depth = distance_transform(mask) * 0.4 + z_rbf_interpolation(pose3d) * 0.6
    """
    N = masks.shape[0]
    _, _, H, W = masks.shape
    masks_np = masks.cpu().squeeze(1).numpy()  # (N, H, W)

    depth_maps = np.zeros((N, H, W), dtype=np.float32)

    try:
        import cv2
    except ImportError:
        logger.warning("cv2 not available for depth generation; returning zero depth maps.")
        return torch.zeros(N, 1, H, W, dtype=torch.float32, device=device)

    for i in range(N):
        mask = (masks_np[i] * 255).astype(np.uint8)

        # Distance transform component
        if mask.max() > 0:
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            dist = dist / (dist.max() + 1e-8)
        else:
            dist = np.zeros((H, W), dtype=np.float32)

        # RBF interpolation of 3D pose z-values
        rbf = _rbf_z_interpolation(kp_2d_np[i], kp_3d_np[i], H, W)

        # Combine
        depth = dist * 0.4 + rbf * 0.6
        depth = depth * masks_np[i]  # mask to mask region

        depth_maps[i] = depth.astype(np.float32)

    depth_tensor = torch.from_numpy(depth_maps).unsqueeze(1).to(device)
    return _ensure_float32(depth_tensor)


def _rbf_z_interpolation(
    kp_2d: np.ndarray,  # (17, 2)
    kp_3d: np.ndarray,  # (17, 3)
    H: int,
    W: int,
    sigma: float = 50.0,
) -> np.ndarray:
    """
    RBF interpolation of 3D pose z-values onto a (H, W) grid.
    Uses inverse distance weighted interpolation.
    """
    grid = np.zeros((H, W), dtype=np.float32)
    z_vals = kp_3d[:, 2]
    valid = (kp_3d[:, 2] != 0.0) | (np.abs(kp_2d[:, 0]) > 0.01) | (np.abs(kp_2d[:, 1]) > 0.01)

    if not valid.any():
        return grid

    valid_pts = kp_2d[valid]       # (M, 2)
    valid_z = z_vals[valid]        # (M,)

    # Coarse sampling for efficiency
    step = max(1, min(H, W) // 64)
    ys, xs = np.mgrid[0:H:step, 0:W:step]

    coarse_grid = np.zeros_like(ys, dtype=np.float32)
    weights_sum = np.zeros_like(ys, dtype=np.float32)

    for j in range(len(valid_pts)):
        dx = xs - valid_pts[j, 0]
        dy = ys - valid_pts[j, 1]
        dist2 = dx * dx + dy * dy
        w = np.exp(-dist2 / (2 * sigma * sigma))
        coarse_grid += w * valid_z[j]
        weights_sum += w

    coarse_grid = np.divide(coarse_grid, weights_sum, out=np.zeros_like(coarse_grid), where=weights_sum > 1e-8)

    # Resize to full resolution
    if step > 1:
        grid = cv2.resize(coarse_grid, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        grid = coarse_grid

    # Normalize to [0, 1]
    gmin = grid.min()
    gmax = grid.max()
    if gmax - gmin > 1e-8:
        grid = (grid - gmin) / (gmax - gmin)

    return grid


# ===================================================================
# 05 — Align Control Signals
# ===================================================================


def align_controls(
    frames: torch.Tensor,
    masks: torch.Tensor,
    heatmaps: torch.Tensor,
    depth_maps: torch.Tensor,
    canny_low: int = 50,
    canny_high: int = 150,
) -> torch.Tensor:
    """
    Generate a 4-channel control pack for consistent video generation.

    Channels:
        R: Canny edges (grayscale frame, masked to mask region)
        G: Distance transform (on mask)
        B: Pose heatmap (aligned)
        A: Temporal difference (absdiff between adjacent frames)

    Args:
        frames:     (N, 3, H, W) float32 [0, 1].
        masks:      (N, 1, H, W) float32 [0, 1].
        heatmaps:   (N, 1, H, W) float32 [0, 1].
        depth_maps: (N, 1, H, W) float32 [0, 1].
        canny_low:  Canny edge low threshold.
        canny_high: Canny edge high threshold.

    Returns:
        control_pack: (N, 4, H, W) float32 [0, 1].
    """
    logger.info(
        "align_controls: %d frames, canny thresholds=(%d, %d)",
        frames.shape[0],
        canny_low,
        canny_high,
    )

    frames = _preprocess_frames(frames)
    N, C, H, W = frames.shape
    device = frames.device

    if N == 0:
        return torch.zeros(0, 4, 1, 1, dtype=torch.float32, device=device)

    # Ensure all inputs are on the same device and have matching spatial dims
    masks = _ensure_float32(masks).to(device)
    heatmaps = _ensure_float32(heatmaps).to(device)
    depth_maps = _ensure_float32(depth_maps).to(device)

    for name, tensor in [("masks", masks), ("heatmaps", heatmaps), ("depth_maps", depth_maps)]:
        if tensor.shape[-2:] != (H, W):
            tensor = F.interpolate(tensor, size=(H, W), mode="bilinear", align_corners=False)
        if tensor.shape[0] == 1 and N > 1:
            tensor = tensor.expand(N, -1, -1, -1)
        if name == "masks":
            masks = tensor
        elif name == "heatmaps":
            heatmaps = tensor
        else:
            depth_maps = tensor

    try:
        import cv2
    except ImportError:
        logger.error("cv2 not available for align_controls.  Returning zero control pack.")
        return torch.zeros(N, 4, H, W, dtype=torch.float32, device=device)

    frames_np = (frames.cpu().numpy() * 255).astype(np.uint8)  # (N, 3, H, W)
    masks_np = masks.cpu().squeeze(1).numpy()  # (N, H, W)
    heatmaps_np = heatmaps.cpu().squeeze(1).numpy()  # (N, H, W)

    control_pack_np = np.zeros((N, 4, H, W), dtype=np.float32)

    for i in range(N):
        # --- R: Canny edges ---
        frame = frames_np[i].transpose(1, 2, 0)  # (H, W, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, canny_low, canny_high)
        edges_f = edges.astype(np.float32) / 255.0
        if i < len(masks_np):
            edges_f = edges_f * masks_np[i]
        control_pack_np[i, 0] = edges_f

        # --- G: Distance transform ---
        if i < len(masks_np):
            mask_u8 = (masks_np[i] * 255).astype(np.uint8)
            if mask_u8.max() > 0:
                dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                dist = dist / (dist.max() + 1e-8)
            else:
                dist = np.zeros((H, W), dtype=np.float32)
        else:
            dist = np.zeros((H, W), dtype=np.float32)
        control_pack_np[i, 1] = dist

        # --- B: Pose heatmap (aligned) ---
        if i < len(heatmaps_np):
            control_pack_np[i, 2] = heatmaps_np[i]
        else:
            control_pack_np[i, 2] = 0.0

        # --- A: Temporal difference ---
        if N > 1:
            if i == 0:
                # First frame: diff with itself or next frame
                curr = frames_np[i].astype(np.float32)
                next_f = frames_np[min(i + 1, N - 1)].astype(np.float32)
                diff = np.abs(curr - next_f)
            else:
                curr = frames_np[i].astype(np.float32)
                prev = frames_np[i - 1].astype(np.float32)
                diff = np.abs(curr - prev)
            diff_gray = diff.mean(axis=0)  # average over channels
            diff_gray = diff_gray / (diff_gray.max() + 1e-8)
            if i < len(masks_np):
                diff_gray = diff_gray * masks_np[i]
            control_pack_np[i, 3] = diff_gray
        else:
            control_pack_np[i, 3] = 0.0

    control_pack = torch.from_numpy(control_pack_np).to(device)
    control_pack = _ensure_float32(control_pack)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    logger.info("align_controls: output control_pack shape %s", control_pack.shape)
    return control_pack


# ===================================================================
# Module-level helper: clear caches
# ===================================================================


def clear_model_cache() -> None:
    """Clear all cached model singletons to free GPU memory."""
    global _MODEL_CACHE, _CACHE_LOCK, _SVG_MASK_DATA
    with _CACHE_LOCK:
        for key, model in _MODEL_CACHE.items():
            if hasattr(model, "cpu"):
                try:
                    model.cpu()
                except Exception:
                    pass
        _MODEL_CACHE.clear()
        _SVG_MASK_DATA.clear()
    logger.info("Model cache cleared.  Freed GPU memory.")


def get_model_cache_info() -> Dict[str, Any]:
    """Return info about cached models."""
    global _MODEL_CACHE
    info: Dict[str, Any] = {}
    for key, model in _MODEL_CACHE.items():
        dev = "unknown"
        try:
            if hasattr(model, "parameters"):
                dev = str(next(model.parameters()).device)
            elif hasattr(model, "device"):
                dev = str(model.device)
        except Exception:
            pass
        info[key] = {"type": type(model).__name__, "device": dev}
    return info