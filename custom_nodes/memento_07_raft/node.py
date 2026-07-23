"""Memento 07 — RAFT 稠密光流时序矫正节点

输入:
  - frames_dir: 01 原始 30fps 帧
  - synthetic_dir: 06 LTX 重绘帧
  - mask_dir: 02 生成的逐帧角色 Mask

输出:
  - flow_aligned_dir: 光流对齐平滑帧（修复帧间漂移和边缘撕裂）

算法:
  1. 对每对帧 (t, t+1):
     a. 计算原始帧 t→t+1 的 RAFT 稠密光流
     b. 计算合成帧 t→t+1 的 RAFT 稠密光流
     c. 计算光流差异 = |原始光流 - 合成光流|
     d. 光流差异超过阈值处，使用原始光流对合成帧进行 warp
     e. Mask 感知混合: Mask 区域内使用 warp 后的合成帧，区域外使用原始帧
  2. RAFT 不可用时回退至 cv2 Farneback 稠密光流
  3. 时序平滑: 对连续 3 帧进行加权平均

当 memento_pipeline.ops.sub.raft_correct 可用时，核心逻辑委托给 tensor-based 实现。
"""

from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Tensor ops 导入 ──
_use_tensor_ops = False
_tensor_raft_correct = None
try:
    from memento_pipeline.ops.sub import raft_correct as _tensor_raft_correct
    _use_tensor_ops = True
    logger.info("[MementoRAFT] 已加载 memento_pipeline.ops.sub.raft_correct，将使用 tensor ops 路径")
except ImportError as e:
    logger.info(f"[MementoRAFT] memento_pipeline.ops.sub 不可用 ({e})，将使用文件级 fallback 路径")

# ── PyTorch + RAFT 导入 ──
_RAFT_AVAILABLE = False
_torch_available = False
try:
    import torch
    _torch_available = True
    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        from torchvision.transforms import functional as F
        _RAFT_AVAILABLE = True
    except ImportError:
        logger.warning("[MementoRAFT] torchvision 未安装 RAFT 模型，将使用 Farneback 回退")
except ImportError:
    logger.warning("[MementoRAFT] PyTorch 未安装，将使用 Farneback 回退")


# ── RAFT 模型单例 ──
class RAFTModel:
    """RAFT 模型加载器（单例模式，GPU 加速）"""

    _instance: Optional["RAFTModel"] = None
    _model = None
    _device = None

    RAFT_MODEL_PATH = os.path.join(os.environ.get("COMFYUI_MODEL_DIR", "/root/data/models"), "raft", "raft_large.pth")

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device

    def load(self) -> object:
        """加载 RAFT 模型（懒加载）"""
        if self._model is not None:
            return self._model

        if not _RAFT_AVAILABLE:
            raise ImportError("RAFT 模型不可用，请安装 torchvision>=0.15")

        start_time = time.time()
        logger.info("[MementoRAFT] 正在加载 RAFT Large 模型...")

        # 尝试从本地路径加载权重
        if os.path.exists(self.RAFT_MODEL_PATH):
            logger.info(f"[MementoRAFT] 从本地路径加载 RAFT: {self.RAFT_MODEL_PATH}")
            try:
                state_dict = torch.load(self.RAFT_MODEL_PATH, map_location=self.device)
                self._model = raft_large(weights=None)
                self._model.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(
                    f"[MementoRAFT] 本地权重加载失败 ({e})，使用预训练权重"
                )
                self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        else:
            logger.info("[MementoRAFT] 本地权重不存在，使用预训练权重")
            self._model = raft_large(weights=Raft_Large_Weights.DEFAULT)

        self._model.to(self.device)
        self._model.eval()

        elapsed = time.time() - start_time
        logger.info(f"[MementoRAFT] RAFT 模型加载完成，耗时 {elapsed:.1f}s，设备: {self.device}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info(f"[MementoRAFT] RAFT 显存占用: {mem_used:.2f} GB")

        return self._model

    def unload(self):
        """释放模型显存"""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[MementoRAFT] RAFT 模型已卸载")


# ── 光流计算工具 ──
class FlowComputer:
    """光流计算器，封装 RAFT 和 Farneback 回退"""

    def __init__(self, use_raft: bool = True):
        self.use_raft = use_raft and _RAFT_AVAILABLE
        self._raft_loader: Optional[RAFTModel] = None
        if self.use_raft:
            self._raft_loader = RAFTModel()
            logger.info("[MementoRAFT] 使用 RAFT 稠密光流模式")
        else:
            logger.info("[MementoRAFT] 使用 Farneback 稠密光流模式（回退）")

    @staticmethod
    def _preprocess_for_raft(img: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        将 BGR uint8 图像预处理为 RAFT 输入格式。
        RAFT 期望输入为 [0, 1] 范围的 RGB float32 tensor，形状 (1, 3, H, W)。
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        return tensor

    def compute_flow_raft(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> np.ndarray:
        """
        使用 RAFT 计算 img_a → img_b 的稠密光流。

        Args:
            img_a: 第一帧 (H, W, 3) BGR uint8
            img_b: 第二帧 (H, W, 3) BGR uint8

        Returns:
            flow: (H, W, 2) float32 光流场，channel 0 = dx, channel 1 = dy
        """
        model = self._raft_loader.load()
        device = self._raft_loader.device
        h, w = img_a.shape[:2]

        # RAFT 需要高度和宽度都是 8 的倍数
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        img_a_pad = cv2.copyMakeBorder(img_a, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        img_b_pad = cv2.copyMakeBorder(img_b, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        tensor_a = self._preprocess_for_raft(img_a_pad, device)
        tensor_b = self._preprocess_for_raft(img_b_pad, device)

        with torch.no_grad():
            flow_list = model(tensor_a, tensor_b)
            flow = flow_list[-1]  # 取最后一层输出的光流，形状 (1, 2, H, W)

        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # 裁掉 padding
        if pad_h > 0 or pad_w > 0:
            flow_np = flow_np[:h, :w, :]

        return flow_np.astype(np.float32)

    def compute_flow_farneback(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> np.ndarray:
        """
        使用 Farneback 算法计算 img_a → img_b 的稠密光流。

        Args:
            img_a: 第一帧 (H, W, 3) BGR uint8
            img_b: 第二帧 (H, W, 3) BGR uint8

        Returns:
            flow: (H, W, 2) float32 光流场
        """
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return flow.astype(np.float32)

    def compute_flow(
        self, img_a: np.ndarray, img_b: np.ndarray
    ) -> np.ndarray:
        """
        计算稠密光流，优先使用 RAFT，回退到 Farneback。

        Args:
            img_a: 第一帧 (H, W, 3) BGR uint8
            img_b: 第二帧 (H, W, 3) BGR uint8

        Returns:
            flow: (H, W, 2) float32 光流场
        """
        if self.use_raft:
            try:
                return self.compute_flow_raft(img_a, img_b)
            except Exception as e:
                logger.warning(f"[MementoRAFT] RAFT 光流计算失败 ({e})，回退 Farneback")
                self.use_raft = False
        return self.compute_flow_farneback(img_a, img_b)


# ── 光流 Warp 工具 ──
def warp_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    使用光流场对图像进行 warp。

    Args:
        img: 输入图像 (H, W, 3) BGR uint8
        flow: 光流场 (H, W, 2) float32

    Returns:
        warped: warp 后的图像 (H, W, 3) BGR uint8
    """
    h, w = flow.shape[:2]
    flow_map = flow.copy()
    # remap 使用 (x, y) 坐标，需要生成网格
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x + flow_map[..., 0]
    map_y = grid_y + flow_map[..., 1]
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """计算光流幅度图"""
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


# ── 主节点 ──
class MementoRAFT:
    """节点 7: RAFT 稠密光流时序矫正

    使用 RAFT 亚像素级稠密光流对齐 LTX 合成帧与原始帧，
    修复帧间漂移和边缘撕裂，并应用时序平滑。
    """

    # 默认光流差异阈值（像素）
    DEFAULT_FLOW_DIFF_THRESHOLD = 2.0

    # 时序平滑窗口大小（奇数）
    TEMPORAL_SMOOTH_WINDOW = 3

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_dir": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "synthetic_dir": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "mask_dir": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "flow_diff_threshold": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.5, "max": 20.0, "step": 0.5},
                ),
                "mask_blend_strength": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "temporal_weight": (
                    "FLOAT",
                    {"default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "use_raft": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "raft_iters": (
                    "INT",
                    {"default": 20, "min": 4, "max": 32, "step": 1,
                     "tooltip": "RAFT 迭代次数，20=推荐，12=快速，32=最高精度"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("flow_aligned_dir",)
    FUNCTION = "correct"
    CATEGORY = "Memento/07_RAFT"

    def __init__(self):
        self._flow_computer: Optional[FlowComputer] = None

    def _load_frame(
        self,
        dir_path: str,
        idx: int,
        target_h: int = 0,
        target_w: int = 0,
        is_gray: bool = False,
        files: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """
        加载并统一尺寸的帧。

        Args:
            dir_path: 帧目录
            idx: 帧索引
            target_h: 目标高度（0 表示不 resize）
            target_w: 目标宽度（0 表示不 resize）
            is_gray: 是否以灰度模式读取
            files: 预排序的文件列表（避免重复 listdir）

        Returns:
            图像 numpy 数组，失败返回 None
        """
        if files is None:
            files = sorted([
                f for f in os.listdir(dir_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])

        if idx >= len(files):
            return None

        flag = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_COLOR
        img = cv2.imread(os.path.join(dir_path, files[idx]), flag)
        if img is None:
            return None

        if target_h > 0 and target_w > 0 and (img.shape[0] != target_h or img.shape[1] != target_w):
            interpolation = cv2.INTER_NEAREST if is_gray else cv2.INTER_LINEAR
            img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

        if not is_gray and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif is_gray and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def _get_frame_info(self, dir_path: str) -> Tuple[int, int, int, List[str]]:
        """
        获取帧序列的尺寸、帧数和文件列表。

        Returns:
            (h, w, num_frames, files)
        """
        files = sorted([
            f for f in os.listdir(dir_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if not files:
            raise RuntimeError(f"帧目录为空: {dir_path}")

        first = cv2.imread(os.path.join(dir_path, files[0]))
        if first is None:
            raise RuntimeError(f"无法读取第一帧: {os.path.join(dir_path, files[0])}")

        h, w = first.shape[:2]
        return h, w, len(files), files

    def _compute_flow_pair(
        self,
        flow_computer: FlowComputer,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> np.ndarray:
        """计算相邻帧对的光流"""
        return flow_computer.compute_flow(img_a, img_b)

    def _apply_correction(
        self,
        original_flow: np.ndarray,
        synthetic_flow: np.ndarray,
        synth_frame: np.ndarray,
        orig_frame: np.ndarray,
        mask: np.ndarray,
        flow_diff_threshold: float,
        mask_blend_strength: float,
    ) -> np.ndarray:
        """
        对单帧应用光流矫正和 Mask 感知混合。

        Args:
            original_flow: 原始帧 t→t+1 的光流
            synthetic_flow: 合成帧 t→t+1 的光流
            synth_frame: 合成帧 t 的图像
            orig_frame: 原始帧 t 的图像
            mask: 当前帧的角色 Mask
            flow_diff_threshold: 光流差异阈值
            mask_blend_strength: Mask 区域内混合强度

        Returns:
            corrected: 矫正后的帧 (H, W, 3) BGR uint8
        """
        h, w = synth_frame.shape[:2]

        # 确保所有输入尺寸一致
        if original_flow.shape[:2] != (h, w):
            original_flow = cv2.resize(original_flow, (w, h))
        if synthetic_flow.shape[:2] != (h, w):
            synthetic_flow = cv2.resize(synthetic_flow, (w, h))
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # 步骤 1: 计算光流差异
        flow_diff = np.abs(original_flow - synthetic_flow)
        flow_diff_mag = flow_magnitude(flow_diff)

        # 步骤 2: 生成光流差异掩码（差异 > 阈值）
        diff_mask = (flow_diff_mag > flow_diff_threshold).astype(np.float32)

        # 高斯模糊差异掩码，避免硬边缘
        diff_mask = cv2.GaussianBlur(diff_mask, (5, 5), 1.0)

        # 步骤 3: 使用原始光流 warp 合成帧
        warped_synth = warp_flow(synth_frame, original_flow)

        # 步骤 4: Mask 感知混合
        # 将 mask 归一化到 [0, 1]
        mask_norm = mask.astype(np.float32) / 255.0
        # 高斯模糊 mask 边缘
        mask_norm = cv2.GaussianBlur(mask_norm, (7, 7), 2.0)

        # 在 Mask 区域内：使用 warp 后的合成帧（按 diff_mask 加权）
        # 在 Mask 区域外：使用原始帧
        # 最终混合公式:
        #   result = (1 - mask_blend) * orig + mask_blend * warped_synth
        #   其中 mask_blend = mask_norm * diff_mask * mask_blend_strength

        blend_weight = mask_norm * diff_mask * mask_blend_strength
        blend_weight = np.clip(blend_weight, 0.0, 1.0)

        # 扩展到 3 通道
        blend_weight_3ch = np.stack([blend_weight] * 3, axis=-1)

        corrected = (
            (1.0 - blend_weight_3ch) * orig_frame.astype(np.float32)
            + blend_weight_3ch * warped_synth.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        return corrected

    def _apply_temporal_smoothing(
        self,
        frame_buffer: List[np.ndarray],
        temporal_weight: float,
    ) -> np.ndarray:
        """
        对帧缓冲区应用时序平滑（加权平均）。

        Args:
            frame_buffer: 连续 3 帧的列表 [prev, curr, next]
            temporal_weight: 相邻帧的权重

        Returns:
            smoothed: 平滑后的当前帧
        """
        if len(frame_buffer) < 3:
            return frame_buffer[-1]

        prev, curr, next_f = frame_buffer
        center_weight = 1.0 - 2.0 * temporal_weight
        center_weight = max(0.0, center_weight)

        smoothed = (
            temporal_weight * prev.astype(np.float32)
            + center_weight * curr.astype(np.float32)
            + temporal_weight * next_f.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        return smoothed

    # ── Tensor Ops 路径 ──
    def _correct_tensor_ops(
        self,
        frames_dir: str,
        synthetic_dir: str,
        mask_dir: str,
        flow_diff_threshold: float,
        mask_blend_strength: float,
        temporal_weight: float,
        use_raft: bool,
    ) -> Tuple[str, int, int, int, float, str]:
        """使用 memento_pipeline.ops.sub.raft_correct 的 tensor ops 路径。"""
        import torch

        logger.info("[MementoRAFT] ====== 使用 Tensor Ops 路径 (memento_pipeline.ops.sub.raft_correct) ======")

        # ── 获取帧信息 ──
        h_frames, w_frames, num_frames, frame_files = self._get_frame_info(frames_dir)
        h_synth, w_synth, num_synth, synth_files = self._get_frame_info(synthetic_dir)
        h_mask, w_mask, num_masks, mask_files = self._get_frame_info(mask_dir)

        num_frames = min(num_frames, num_synth, num_masks)
        target_h, target_w = h_frames, w_frames

        logger.info(f"[MementoRAFT] 对齐后帧数: {num_frames}, 分辨率: {target_w}x{target_h}")

        if num_frames < 2:
            raise RuntimeError(f"帧数不足（需要至少 2 帧）: {num_frames}")

        # ── 加载帧到 numpy 数组 ──
        logger.info("[MementoRAFT] 预加载帧到内存...")
        orig_frames_np = []
        synth_frames_np = []
        masks_np = []

        for i in range(num_frames):
            orig = self._load_frame(frames_dir, i, target_h, target_w, is_gray=False, files=frame_files)
            synth = self._load_frame(synthetic_dir, i, target_h, target_w, is_gray=False, files=synth_files)
            msk = self._load_frame(mask_dir, i, target_h, target_w, is_gray=True, files=mask_files)

            if orig is None or synth is None or msk is None:
                raise RuntimeError(f"帧 {i+1} 加载失败")

            orig_frames_np.append(orig)
            synth_frames_np.append(synth)
            masks_np.append(msk)

        # ── 转换为 tensor ──
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # numpy BGR uint8 -> tensor RGB float32 [0,1] (N, 3, H, W)
        orig_tensor = torch.from_numpy(
            np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in orig_frames_np]).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).to(device)

        synth_tensor = torch.from_numpy(
            np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in synth_frames_np]).astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2).to(device)

        # mask: (N, H, W) uint8 -> (N, 1, H, W) float32 [0,1]
        mask_tensor = torch.from_numpy(
            np.stack(masks_np).astype(np.float32) / 255.0
        ).unsqueeze(1).to(device)

        logger.info(
            f"[MementoRAFT] 张量准备完成: orig={orig_tensor.shape}, "
            f"synth={synth_tensor.shape}, mask={mask_tensor.shape}, device={device}"
        )

        # ── 调用 tensor ops ──
        total_start = time.time()
        aligned_tensor = _tensor_raft_correct(
            original_frames=orig_tensor,
            synthetic_frames=synth_tensor,
            masks=mask_tensor,
            flow_diff_threshold=flow_diff_threshold,
            mask_blend_strength=mask_blend_strength,
        )  # (N, 3, H, W) float32 [0,1]

        total_elapsed = time.time() - total_start

        # ── 保存输出帧 ──
        flow_aligned_dir = "/workspace/flow_aligned"
        Path(flow_aligned_dir).mkdir(parents=True, exist_ok=True)

        aligned_np = aligned_tensor.cpu().permute(0, 2, 3, 1).numpy()  # (N, H, W, 3) float32 [0,1]
        # RGB -> BGR
        aligned_np = (aligned_np * 255.0).clip(0, 255).astype(np.uint8)
        aligned_np = aligned_np[..., ::-1].copy()

        logger.info(f"[MementoRAFT] 保存矫正帧到 {flow_aligned_dir}...")
        for i, frame in enumerate(aligned_np):
            out_path = os.path.join(flow_aligned_dir, f"aligned_{i+1:05d}.png")
            cv2.imwrite(out_path, frame)

        fps_processing = num_frames / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"[MementoRAFT] Tensor ops 处理完成: {num_frames} 帧, "
            f"耗时 {total_elapsed:.1f}s ({fps_processing:.2f} fps)"
        )

        return flow_aligned_dir, num_frames, target_w, target_h, total_elapsed, fps_processing

    # ── 文件级 Fallback 路径 ──
    def _correct_file_based(
        self,
        frames_dir: str,
        synthetic_dir: str,
        mask_dir: str,
        flow_diff_threshold: float,
        mask_blend_strength: float,
        temporal_weight: float,
        use_raft: bool,
    ) -> Tuple[str, int, int, int, float, float]:
        """文件级 fallback 路径（原有逻辑）。"""
        logger.info("[MementoRAFT] ====== 使用文件级 Fallback 路径 ======")

        # ── 获取各目录帧信息 ──
        h_frames, w_frames, num_frames, frame_files = self._get_frame_info(frames_dir)
        h_synth, w_synth, num_synth, synth_files = self._get_frame_info(synthetic_dir)
        h_mask, w_mask, num_masks, mask_files = self._get_frame_info(mask_dir)

        logger.info(f"[MementoRAFT] 原始帧: {num_frames} 帧, {w_frames}x{h_frames}")
        logger.info(f"[MementoRAFT] 合成帧: {num_synth} 帧, {w_synth}x{h_synth}")
        logger.info(f"[MementoRAFT] Mask: {num_masks} 帧, {w_mask}x{h_mask}")

        # 取最小帧数对齐
        num_frames = min(num_frames, num_synth, num_masks)
        logger.info(f"[MementoRAFT] 对齐后帧数: {num_frames}")

        if num_frames < 2:
            raise RuntimeError(f"帧数不足（需要至少 2 帧）: {num_frames}")

        # 统一分辨率为原始帧尺寸
        target_h, target_w = h_frames, w_frames

        # ── 创建输出目录 ──
        flow_aligned_dir = "/workspace/flow_aligned"
        Path(flow_aligned_dir).mkdir(parents=True, exist_ok=True)

        # ── 初始化光流计算器 ──
        flow_computer = FlowComputer(use_raft=use_raft)

        # ── 逐帧处理 ──
        total_start = time.time()

        # 时序平滑缓冲区: [prev, curr, next]
        temporal_buffer: List[np.ndarray] = []

        # 先加载所有帧到内存（批量处理优化）
        logger.info("[MementoRAFT] 预加载帧到内存...")
        orig_frames: List[np.ndarray] = []
        synth_frames: List[np.ndarray] = []
        masks: List[np.ndarray] = []

        for i in range(num_frames):
            orig = self._load_frame(
                frames_dir, i, target_h, target_w, is_gray=False, files=frame_files
            )
            synth = self._load_frame(
                synthetic_dir, i, target_h, target_w, is_gray=False, files=synth_files
            )
            msk = self._load_frame(
                mask_dir, i, target_h, target_w, is_gray=True, files=mask_files
            )

            if orig is None or synth is None or msk is None:
                raise RuntimeError(f"帧 {i+1} 加载失败")

            orig_frames.append(orig)
            synth_frames.append(synth)
            masks.append(msk)

        logger.info(
            f"[MementoRAFT] 预加载完成: {num_frames} 帧, "
            f"内存约 {num_frames * target_h * target_w * 3 * 3 / (1024**2):.1f} MB"
        )

        # ── 计算光流、矫正、时序平滑 ──
        corrected_frames: List[np.ndarray] = []

        for i in range(num_frames):
            # 最后一帧没有 t+1，直接使用合成帧
            if i == num_frames - 1:
                corrected = synth_frames[i].copy()
                corrected_frames.append(corrected)
                logger.info(
                    f"[MementoRAFT] 帧 {i+1}/{num_frames}: 最后一帧，直接使用合成帧"
                )
                continue

            # 计算原始光流 t→t+1
            orig_flow = self._compute_flow_pair(
                flow_computer, orig_frames[i], orig_frames[i + 1]
            )

            # 计算合成光流 t→t+1
            synth_flow = self._compute_flow_pair(
                flow_computer, synth_frames[i], synth_frames[i + 1]
            )

            # 应用矫正
            corrected = self._apply_correction(
                orig_flow,
                synth_flow,
                synth_frames[i],
                orig_frames[i],
                masks[i],
                flow_diff_threshold,
                mask_blend_strength,
            )

            corrected_frames.append(corrected)

            if (i + 1) % 30 == 0 or i == num_frames - 2:
                logger.info(
                    f"[MementoRAFT] 光流矫正进度: {i+1}/{num_frames} 帧"
                )

        # ── 时序平滑 ──
        logger.info("[MementoRAFT] 应用时序平滑...")
        final_frames: List[np.ndarray] = []

        for i in range(num_frames):
            if i == 0:
                # 第一帧: 使用当前帧和下一帧的平均
                prev = corrected_frames[i]
                curr = corrected_frames[i]
                next_f = corrected_frames[i + 1] if i + 1 < num_frames else curr
            elif i == num_frames - 1:
                # 最后一帧: 使用当前帧和上一帧的平均
                prev = corrected_frames[i - 1]
                curr = corrected_frames[i]
                next_f = curr
            else:
                prev = corrected_frames[i - 1]
                curr = corrected_frames[i]
                next_f = corrected_frames[i + 1]

            smoothed = self._apply_temporal_smoothing(
                [prev, curr, next_f], temporal_weight
            )
            final_frames.append(smoothed)

        # ── 保存输出帧 ──
        logger.info(f"[MementoRAFT] 保存矫正帧到 {flow_aligned_dir}...")
        for i, frame in enumerate(final_frames):
            out_path = os.path.join(
                flow_aligned_dir, f"aligned_{i+1:05d}.png"
            )
            cv2.imwrite(out_path, frame)

        total_elapsed = time.time() - total_start
        fps_processing = num_frames / total_elapsed if total_elapsed > 0 else 0

        logger.info(
            f"[MementoRAFT] 处理完成: {num_frames} 帧, "
            f"耗时 {total_elapsed:.1f}s ({fps_processing:.2f} fps)"
        )

        # 释放 RAFT 模型显存
        if flow_computer.use_raft and flow_computer._raft_loader is not None:
            flow_computer._raft_loader.unload()

        return flow_aligned_dir, num_frames, target_w, target_h, total_elapsed, fps_processing

    # ── 主入口 ──
    def correct(
        self,
        frames_dir: str,
        synthetic_dir: str,
        mask_dir: str,
        flow_diff_threshold: float,
        mask_blend_strength: float,
        temporal_weight: float,
        use_raft: bool,
    ) -> Tuple[str]:
        logger.info(
            f"[MementoRAFT] 开始稠密光流时序矫正:\n"
            f"  frames_dir={frames_dir}\n"
            f"  synthetic_dir={synthetic_dir}\n"
            f"  mask_dir={mask_dir}\n"
            f"  flow_diff_threshold={flow_diff_threshold}\n"
            f"  mask_blend_strength={mask_blend_strength}\n"
            f"  temporal_weight={temporal_weight}\n"
            f"  use_raft={use_raft}\n"
            f"  _use_tensor_ops={_use_tensor_ops}"
        )

        # ── 输入验证 ──
        for path, name in [
            (frames_dir, "frames_dir"),
            (synthetic_dir, "synthetic_dir"),
            (mask_dir, "mask_dir"),
        ]:
            if not path or not os.path.exists(path):
                raise FileNotFoundError(f"{name} 目录不存在: {path}")

        # ── 选择路径 ──
        if _use_tensor_ops and _tensor_raft_correct is not None:
            try:
                flow_aligned_dir, num_frames, target_w, target_h, total_elapsed, fps_processing = (
                    self._correct_tensor_ops(
                        frames_dir, synthetic_dir, mask_dir,
                        flow_diff_threshold, mask_blend_strength,
                        temporal_weight, use_raft,
                    )
                )
            except Exception as e:
                logger.warning(
                    f"[MementoRAFT] Tensor ops 路径失败 ({e})，回退到文件级 fallback"
                )
                flow_aligned_dir, num_frames, target_w, target_h, total_elapsed, fps_processing = (
                    self._correct_file_based(
                        frames_dir, synthetic_dir, mask_dir,
                        flow_diff_threshold, mask_blend_strength,
                        temporal_weight, use_raft,
                    )
                )
        else:
            flow_aligned_dir, num_frames, target_w, target_h, total_elapsed, fps_processing = (
                self._correct_file_based(
                    frames_dir, synthetic_dir, mask_dir,
                    flow_diff_threshold, mask_blend_strength,
                    temporal_weight, use_raft,
                )
            )

        # ── 更新 context.json ──
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            try:
                with open(context_path, "r") as f:
                    context = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"[MementoRAFT] 读取 context.json 失败: {e}")

        context.update({
            "flow_aligned_dir": flow_aligned_dir,
            "num_flow_aligned_frames": num_frames,
            "flow_diff_threshold": flow_diff_threshold,
            "mask_blend_strength": mask_blend_strength,
            "temporal_weight": temporal_weight,
            "flow_method": "tensor_ops" if _use_tensor_ops else ("RAFT" if use_raft else "Farneback"),
            "flow_aligned_width": target_w,
            "flow_aligned_height": target_h,
            "processing_time_sec": round(total_elapsed, 1),
            "processing_fps": round(fps_processing, 2),
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(
            f"[MementoRAFT] 稠密光流时序矫正完成, 输出到 {flow_aligned_dir}"
        )
        return (flow_aligned_dir,)


NODE_CLASS_MAPPINGS = {"MementoRAFT": MementoRAFT}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoRAFT": "Memento 07 - RAFT 稠密光流时序矫正"}