"""
Memento 08 — Layered Lighting/Color Fusion Node (无冗余背景)
===============================================================

输入:
  - flow_aligned_dir: 07 光流矫正帧 (来自 RAFT 矫正，已保留背景)
  - mask_dir:         02 分层遮罩 (来自 SAM3，4 层 SVG mask)
  - depth_dir:        04 深度图 (来自 MotionBERT)

说明:
  - 07 光流矫正帧已保留原始背景，无需额外传入 01 原始背景素材
  - 融合策略: 4 层遮罩 + 颜色匹配 + 深度感知阴影调整

4 层遮罩策略:
  Layer 0 (foreground) : 直接替换 (mask > 0.5)
  Layer 1 (feather)    : 高斯模糊 alpha 混合
  Layer 2 (detail)     : 拉普拉斯金字塔边缘保持混合
  Layer 3 (semitrans)  : Screen 混合模式

颜色匹配: 直方图匹配前景区到背景边界区域
深度感知阴影: 深度越大 → 阴影越暗 (0.5~1.0 倍乘)
"""

import os
import glob
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── 尝试导入 memento_pipeline.ops.sub GPU 张量操作 ──
try:
    from memento_pipeline.ops.sub import fusion_blend as _ops_fusion_blend
    _TENSOR_OPS_AVAILABLE = True
    logger.info("[MementoFusion] memento_pipeline.ops.sub 已加载，将使用 GPU 张量操作")
except ImportError:
    _TENSOR_OPS_AVAILABLE = False
    logger.info("[MementoFusion] memento_pipeline.ops.sub 未安装，使用文件级回退逻辑")


class MementoFusion:
    """ComfyUI custom node: Memento 08 — 分层光影融合 (无冗余背景)

    输入: 07 光流矫正帧 + 02 分层遮罩 + 04 深度图
    (无需原始背景，07 已保留)

    融合 07 光流矫正帧，使用 4 层遮罩策略:
      Layer 0 (foreground) : direct replacement
      Layer 1 (feather)    : alpha blending with gaussian weight
      Layer 2 (detail)     : edge-preserving Poisson-style blending
      Layer 3 (semitrans)  : screen blend mode

    颜色匹配和深度感知阴影调整在最终合成前应用。
    """

    # 是否使用 GPU 张量操作
    _use_tensor_ops = _TENSOR_OPS_AVAILABLE

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow_aligned_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to 07 optical-flow corrected frames"
                }),
                "mask_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to 02 masks (4-layer SVG mask if available)"
                }),
                "depth_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to 04 depth maps"
                }),
                "blend_alpha": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "融合透明度，Result = FG*alpha + BG*(1-alpha)",
                }),
                "feather_radius": ("INT", {
                    "default": 10, "min": 0, "max": 30, "step": 1,
                    "tooltip": "边缘羽化半径(像素)，0=硬边缘，10=推荐",
                }),
                "shadow_strength": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "深度阴影强度，0.3-0.5=自然阴影",
                }),
                "color_match_strength": ("FLOAT", {
                    "default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "颜色匹配强度，0.5-0.8=推荐",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_frames_dir",)
    FUNCTION = "blend"
    CATEGORY = "Memento/08_Fusion"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sorted_image_files(directory):
        """Return a sorted list of absolute image file paths from *directory*."""
        if not directory or not os.path.isdir(directory):
            return []
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp", "*.exr")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(directory, ext)))
        files.sort()
        return files

    @staticmethod
    def _load_image(path, grayscale=False):
        """Load an image from disk.  Returns None on failure."""
        if not os.path.isfile(path):
            return None
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        if img is None:
            return None
        if not grayscale and img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif grayscale and img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return img

    @staticmethod
    def _save_image(path, img):
        """Save a float [0,1] BGR image to disk as 8-bit PNG."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(path, out)

    # ------------------------------------------------------------------
    # Mask handling
    # ------------------------------------------------------------------

    @staticmethod
    def _load_mask(frame_index, mask_paths, h, w):
        """Load the 4-layer mask for a given frame.

        If *mask_paths* is a list of per-frame mask files, use the
        corresponding index.  If only one mask file exists, reuse it
        (static mask).  The mask is expected to be a 4-channel image
        where each channel encodes one layer:
          ch0 → foreground
          ch1 → feather
          ch2 → detail
          ch3 → semitransparent

        Returns a dict of float32 masks in [0,1], resized to (h,w).
        Returns None if no mask is available.
        """
        if not mask_paths:
            return None

        path = mask_paths[min(frame_index, len(mask_paths) - 1)]
        mask_img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            return None

        # Resize to match working resolution
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_LINEAR)

        if mask_img.dtype == np.uint8:
            mask_img = mask_img.astype(np.float32) / 255.0

        if len(mask_img.shape) == 2:
            # Single-channel mask: treat as foreground-only
            return {
                "foreground": mask_img,
                "feather": np.zeros_like(mask_img),
                "detail": np.zeros_like(mask_img),
                "semitrans": np.zeros_like(mask_img),
            }

        ch = mask_img.shape[2]
        return {
            "foreground": mask_img[:, :, 0] if ch >= 1 else np.zeros((h, w), dtype=np.float32),
            "feather":    mask_img[:, :, 1] if ch >= 2 else np.zeros((h, w), dtype=np.float32),
            "detail":     mask_img[:, :, 2] if ch >= 3 else np.zeros((h, w), dtype=np.float32),
            "semitrans":  mask_img[:, :, 3] if ch >= 4 else np.zeros((h, w), dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Layer blending strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_foreground(fg, bg, mask):
        """Layer 0: Direct replacement where mask > 0.5."""
        m = (mask > 0.5).astype(np.float32)
        m3 = np.dstack([m, m, m])
        return fg * m3 + bg * (1.0 - m3)

    @staticmethod
    def _blend_feather(fg, bg, mask):
        """Layer 1: Alpha blending with Gaussian-smoothed mask."""
        m = cv2.GaussianBlur(mask, (21, 21), 7)
        m3 = np.dstack([m, m, m])
        return fg * m3 + bg * (1.0 - m3)

    @staticmethod
    def _blend_detail(fg, bg, mask):
        """Layer 2: Edge-preserving Poisson-style blending.

        Uses a Laplace-pyramid reconstruction on the foreground,
        blended with the background at the mask boundary.
        This approximates Poisson image editing.
        """
        h, w = mask.shape
        m = cv2.GaussianBlur(mask, (5, 5), 2)
        m3 = np.dstack([m, m, m])

        # Build Laplacian pyramids
        depth = 4
        gauss_fg = [fg.copy()]
        gauss_bg = [bg.copy()]
        gauss_m  = [m3.copy()]

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

        # Composite each pyramid level
        composite = []
        for i in range(depth + 1):
            hh, ww = laplace_fg[i].shape[:2]
            gm = cv2.resize(gauss_m[min(i, depth)], (ww, hh))
            if len(gm.shape) == 2:
                gm = np.dstack([gm, gm, gm])
            lvl = laplace_fg[i] * gm + laplace_bg[i] * (1.0 - gm)
            composite.append(lvl)

        # Reconstruct
        result = composite[-1]
        for i in range(depth - 1, -1, -1):
            result = cv2.pyrUp(result)
            hh, ww = composite[i].shape[:2]
            result = cv2.resize(result, (ww, hh))
            result += composite[i]

        return np.clip(result, 0, 1)

    @staticmethod
    def _blend_semitrans(fg, bg, mask):
        """Layer 3: Screen blend mode for semi-transparent regions."""
        m = cv2.GaussianBlur(mask, (11, 11), 4)
        m3 = np.dstack([m, m, m])

        # Screen blend: 1 - (1-a)*(1-b)
        screen = 1.0 - (1.0 - fg) * (1.0 - bg)
        return fg * m3 + screen * (1.0 - m3)

    # ------------------------------------------------------------------
    # Colour matching
    # ------------------------------------------------------------------

    @staticmethod
    def _histogram_match(source, target):
        """Match the histogram of *source* to *target* (both BGR float [0,1])."""
        result = np.zeros_like(source)
        for c in range(3):
            src_ch = (source[:, :, c] * 255).astype(np.uint8)
            tgt_ch = (target[:, :, c] * 255).astype(np.uint8)

            src_hist, _ = np.histogram(src_ch, 256, [0, 256])
            tgt_hist, _ = np.histogram(tgt_ch, 256, [0, 256])

            src_cdf = np.cumsum(src_hist).astype(np.float64)
            tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)

            src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
            tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1

            lut = np.zeros(256, dtype=np.uint8)
            tj = 0
            for i in range(256):
                while tj < 256 and tgt_cdf[tj] < src_cdf[i]:
                    tj += 1
                lut[i] = tj

            matched = lut[src_ch]
            result[:, :, c] = matched.astype(np.float32) / 255.0

        return result

    @staticmethod
    def _color_match(fg, bg, mask, depth_map=None):
        """Match foreground colour distribution to the background region
        near the mask boundary, then apply depth-based shadow adjustment.

        Returns colour-corrected foreground.
        """
        if mask is None or mask.max() < 1e-6:
            return fg

        h, w = mask.shape

        # Dilate the mask to get the boundary region on the background side
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated = cv2.dilate(mask, kernel, iterations=3)
        eroded  = cv2.erode(mask, kernel, iterations=3)
        boundary = np.clip(dilated - eroded, 0, 1)

        # Background region near the boundary
        bg_boundary = bg * np.dstack([boundary, boundary, boundary])

        # Only use pixels that actually have content
        valid = boundary > 0.01
        if valid.sum() < 10:
            return fg

        # Histogram match foreground to the boundary background
        corrected = MementoFusion._histogram_match(fg, bg_boundary)

        # Depth-based shadow adjustment
        if depth_map is not None:
            depth = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            if depth.dtype == np.uint8:
                depth = depth.astype(np.float32) / 255.0
            # Normalize depth to [0,1] range
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)

            # Deeper regions get darker (shadow intensity 0.5 – 1.0)
            shadow = 1.0 - 0.5 * depth  # range [0.5, 1.0]
            shadow3 = np.dstack([shadow, shadow, shadow])

            # Apply shadow only where mask is active
            m3 = np.dstack([mask, mask, mask])
            corrected = corrected * (shadow3 * m3 + (1.0 - m3))

        return np.clip(corrected, 0, 1)

    # ------------------------------------------------------------------
    # GPU 张量操作路径
    # ------------------------------------------------------------------

    def _blend_tensor_ops(self, flow_aligned_dir, mask_dir, depth_dir):
        """使用 memento_pipeline.ops.sub.fusion_blend 进行 GPU 张量操作"""
        if not _TENSOR_OPS_AVAILABLE:
            raise RuntimeError("memento_pipeline.ops.sub 不可用，无法使用张量操作路径")

        import torch

        aligned_paths = self._sorted_image_files(flow_aligned_dir)
        mask_paths = self._sorted_image_files(mask_dir)
        depth_paths = self._sorted_image_files(depth_dir)

        if not aligned_paths:
            raise ValueError(
                f"No image files found in flow_aligned_dir: {flow_aligned_dir}"
            )

        num_frames = len(aligned_paths)
        logger.info(f"[MementoFusion] (tensor ops) Found {num_frames} aligned frames")

        # 加载第一帧获取尺寸
        first = self._load_image(aligned_paths[0])
        if first is None:
            raise RuntimeError(f"无法读取第一帧: {aligned_paths[0]}")
        H, W = first.shape[:2]

        # 加载所有帧为张量 (N, 3, H, W) float32 [0, 1]
        frames_np = []
        for i in range(num_frames):
            fg = self._load_image(aligned_paths[i])
            if fg is None:
                fg = np.zeros((H, W, 3), dtype=np.float32)
            elif fg.shape[:2] != (H, W):
                fg = cv2.resize(fg, (W, H))
            frames_np.append(fg)
        frames_np = np.stack(frames_np, axis=0)  # (N, H, W, 3)
        frames_t = torch.from_numpy(frames_np.astype(np.float32)).permute(0, 3, 1, 2)  # (N, 3, H, W)

        # 加载所有掩码为张量 (N, 1, H, W) float32 [0, 1]
        masks_np = np.zeros((num_frames, H, W), dtype=np.float32)
        for i in range(min(num_frames, len(mask_paths))):
            m = self._load_image(mask_paths[i], grayscale=True)
            if m is not None:
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H))
                masks_np[i] = m
        masks_t = torch.from_numpy(masks_np).unsqueeze(1)  # (N, 1, H, W)

        # 加载所有深度图为张量 (N, 1, H, W) float32 [0, 1]
        depth_np = np.zeros((num_frames, H, W), dtype=np.float32)
        for i in range(min(num_frames, len(depth_paths))):
            d = self._load_image(depth_paths[i], grayscale=True)
            if d is not None:
                if d.shape[:2] != (H, W):
                    d = cv2.resize(d, (W, H))
                depth_np[i] = d
        depth_t = torch.from_numpy(depth_np).unsqueeze(1)  # (N, 1, H, W)

        # 调用 ops.sub.fusion_blend
        final_frames_t = _ops_fusion_blend(
            synthetic_frames=frames_t,
            masks=masks_t,
            depth_maps=depth_t,
        )  # (N, 3, H, W) float32 [0, 1]

        # 创建输出目录
        base_out = os.path.dirname(os.path.normpath(flow_aligned_dir))
        output_dir = os.path.join(base_out, "08_final_frames")
        os.makedirs(output_dir, exist_ok=True)

        # 保存结果
        final_np = final_frames_t.cpu().numpy()  # (N, 3, H, W)
        for i in range(num_frames):
            frame = final_np[i].transpose(1, 2, 0)  # (H, W, 3)
            frame = np.clip(frame, 0, 1)
            out_path = os.path.join(output_dir, f"frame_{i:06d}.png")
            self._save_image(out_path, frame)

            if (i + 1) % max(1, num_frames // 10) == 0 or i == num_frames - 1:
                logger.info(f"[MementoFusion] (tensor ops) Processed {i + 1}/{num_frames} frames")

        logger.info(f"[MementoFusion] (tensor ops) Done. Output: {output_dir}")
        return (output_dir,)

    # ------------------------------------------------------------------
    # 文件级回退路径
    # ------------------------------------------------------------------

    def _blend_file_based(self, flow_aligned_dir, mask_dir, depth_dir):
        """使用文件级逻辑进行分层光影融合（无冗余背景）"""
        # --- Gather input files ---
        aligned_paths = self._sorted_image_files(flow_aligned_dir)
        mask_paths    = self._sorted_image_files(mask_dir)
        depth_paths   = self._sorted_image_files(depth_dir)

        if not aligned_paths:
            raise ValueError(
                f"No image files found in flow_aligned_dir: {flow_aligned_dir}"
            )

        num_frames = len(aligned_paths)

        logger.info(f"[MementoFusion] (file-based) Found {num_frames} aligned frames")
        logger.info(f"[MementoFusion] (file-based) Found {len(mask_paths)} mask files")
        logger.info(f"[MementoFusion] (file-based) Found {len(depth_paths)} depth maps")

        # --- Create output directory ---
        base_out = os.path.dirname(os.path.normpath(flow_aligned_dir))
        output_dir = os.path.join(base_out, "08_final_frames")
        os.makedirs(output_dir, exist_ok=True)

        # --- Process each frame ---
        for idx in range(num_frames):
            # Load foreground (aligned) — 07 输出已保留背景
            fg = self._load_image(aligned_paths[idx])
            if fg is None:
                logger.warning(f"[MementoFusion] WARNING: skipping frame {idx} — "
                              f"cannot load {aligned_paths[idx]}")
                continue

            h, w = fg.shape[:2]

            # 07 输出已保留背景，bg 就是 fg 本身
            bg = fg.copy()

            # Load depth map
            depth_map = None
            if depth_paths:
                depth_map = self._load_image(
                    depth_paths[min(idx, len(depth_paths) - 1)],
                    grayscale=True,
                )
                if depth_map is not None and depth_map.shape[:2] != (h, w):
                    depth_map = cv2.resize(
                        depth_map, (w, h), interpolation=cv2.INTER_LINEAR,
                    )

            # Load mask
            layers = self._load_mask(idx, mask_paths, h, w)

            # --- Layer 0: Foreground direct replacement ---
            if layers is not None and layers["foreground"].max() > 0.01:
                composite = self._blend_foreground(fg, bg, layers["foreground"])
            else:
                composite = fg.copy()

            # --- Colour matching ---
            if layers is not None:
                combined_mask = np.clip(
                    layers["foreground"]
                    + layers["feather"]
                    + layers["detail"]
                    + layers["semitrans"],
                    0, 1,
                )
            else:
                combined_mask = np.ones((h, w), dtype=np.float32)

            fg_matched = self._color_match(fg, bg, combined_mask, depth_map)

            # --- Layer 1: Feather blending ---
            if layers is not None and layers["feather"].max() > 0.01:
                composite = self._blend_feather(fg_matched, composite, layers["feather"])

            # --- Layer 2: Detail (edge-preserving) ---
            if layers is not None and layers["detail"].max() > 0.01:
                composite = self._blend_detail(fg_matched, composite, layers["detail"])

            # --- Layer 3: Semitransparent screen blend ---
            if layers is not None and layers["semitrans"].max() > 0.01:
                composite = self._blend_semitrans(fg_matched, composite, layers["semitrans"])

            # --- Final composite: overlay foreground on background ---
            if layers is not None:
                final_mask = np.clip(
                    layers["foreground"]
                    + layers["feather"]
                    + layers["detail"]
                    + layers["semitrans"],
                    0, 1,
                )
                m3 = np.dstack([final_mask, final_mask, final_mask])
                final = composite * m3 + bg * (1.0 - m3)
            else:
                final = composite

            # Save
            out_path = os.path.join(output_dir, f"frame_{idx:06d}.png")
            self._save_image(out_path, final)

            if (idx + 1) % max(1, num_frames // 10) == 0 or idx == num_frames - 1:
                logger.info(f"[MementoFusion] (file-based) Processed {idx + 1}/{num_frames} frames")

        logger.info(f"[MementoFusion] (file-based) Done. Output: {output_dir}")
        return (output_dir,)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def blend(self, flow_aligned_dir, mask_dir, depth_dir):
        """Run the full fusion pipeline.

        输入: 07 光流矫正帧 + 02 分层遮罩 + 04 深度图
        (无需原始背景，07 已保留)

        Returns the path to the directory containing final composite frames.
        """
        logger.info(
            f"[MementoFusion] blend: flow_aligned={flow_aligned_dir}, "
            f"mask={mask_dir}, depth={depth_dir}, "
            f"tensor_ops={self._use_tensor_ops}"
        )

        if self._use_tensor_ops and _TENSOR_OPS_AVAILABLE:
            try:
                return self._blend_tensor_ops(flow_aligned_dir, mask_dir, depth_dir)
            except Exception as e:
                logger.warning(
                    f"[MementoFusion] GPU 张量操作失败: {e}，回退到文件级逻辑"
                )

        return self._blend_file_based(flow_aligned_dir, mask_dir, depth_dir)


# ------------------------------------------------------------------
# ComfyUI registration
# ------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MementoFusion": MementoFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MementoFusion": "Memento 08 - 分层光影融合 (无冗余背景)",
}