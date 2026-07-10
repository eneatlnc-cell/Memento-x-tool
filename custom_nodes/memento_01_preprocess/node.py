"""Memento 01 — FFmpeg 预处理节点

将 H.264/HEVC 视频:
  1. 拆帧为 30fps 原始画面帧 (PNG)
  2. 分离独立原始音频文件 (WAV)
  3. ffprobe 提取分辨率/帧率/时长/色彩元数据

写入视频元数据到 /workspace/context.json
"""
import logging
import json
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MementoPreprocess:
    """节点 1: 视频预处理 — 拆帧 + 音频分离 + 元数据提取"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "max_resolution": (["1080p", "4K", "8K"], {"default": "1080p"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("frames_dir", "audio_path", "frame_count", "metadata_json")
    FUNCTION = "process"
    CATEGORY = "Memento/01_Preprocess"

    # ── 分辨率映射 ──
    RES_LIMITS = {"1080p": 1920, "4K": 3840, "8K": 7680}

    def get_video_metadata(self, video_path: str) -> dict:
        """
        用 ffprobe 获取完整视频元数据:
        - 视频流: width, height, duration, nb_frames, r_frame_rate, pix_fmt, color_space, color_transfer, color_primaries
        - 音频流: codec_name, sample_rate, channels
        """
        cmd = [
            "ffprobe", "-v", "error",
            "-show_streams",
            "-show_format",
            "-of", "json",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败: {result.stderr}")

        data = json.loads(result.stdout)

        # 找视频流
        video_stream = None
        audio_stream = None
        for s in data.get("streams", []):
            if s["codec_type"] == "video" and video_stream is None:
                video_stream = s
            elif s["codec_type"] == "audio" and audio_stream is None:
                audio_stream = s

        if video_stream is None:
            raise RuntimeError("未找到视频流")

        # 解析帧率 (可能是分数如 "30000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = round(float(num) / float(den), 2)

        meta = {
            "width": int(video_stream["width"]),
            "height": int(video_stream["height"]),
            "duration": float(data.get("format", {}).get("duration", "0")),
            "nb_frames": int(video_stream.get("nb_read_frames", "0")),
            "fps": fps,
            "pix_fmt": video_stream.get("pix_fmt", "unknown"),
            "color_space": video_stream.get("color_space", "unknown"),
            "color_transfer": video_stream.get("color_transfer", "unknown"),
            "color_primaries": video_stream.get("color_primaries", "unknown"),
            "codec": video_stream.get("codec_name", "unknown"),
            "bit_rate": int(data.get("format", {}).get("bit_rate", "0")),
        }

        if audio_stream:
            meta["audio"] = {
                "codec": audio_stream.get("codec_name", "unknown"),
                "sample_rate": int(audio_stream.get("sample_rate", "0")),
                "channels": int(audio_stream.get("channels", "0")),
            }

        return meta

    def extract_audio(self, video_path: str, output_dir: str) -> str:
        """从视频中分离独立原始音频 (WAV 16bit PCM)"""
        audio_path = os.path.join(output_dir, "original_audio.wav")

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",                          # 不要视频
            "-acodec", "pcm_s16le",         # 16bit PCM WAV
            "-ar", "48000",                 # 48kHz
            "-ac", "2",                     # 立体声
            audio_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"[MementoPreprocess] 音频分离失败（可能无音轨）: {result.stderr}")
            # 无音轨不是致命错误，跳过
            return ""

        logger.info(f"[MementoPreprocess] 音频已分离: {audio_path}")
        return audio_path

    def process(self, video_path: str, output_fps: int, max_resolution: str):
        logger.info(
            f"[MementoPreprocess] 输入: {video_path}, fps={output_fps}, "
            f"res={max_resolution}"
        )

        # 检查输入文件存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入视频不存在: {video_path}")

        # 创建输出目录
        frames_dir = "/workspace/frames"
        Path(frames_dir).mkdir(parents=True, exist_ok=True)

        # ── 获取原视频元数据 ──
        meta = self.get_video_metadata(video_path)
        logger.info(
            f"[MementoPreprocess] 原视频: {meta['width']}x{meta['height']}, "
            f"{meta['fps']}fps, {meta['nb_frames']}帧, "
            f"pix_fmt={meta['pix_fmt']}, color={meta['color_space']}"
        )

        # ── 分离音频 ──
        audio_path = self.extract_audio(video_path, frames_dir)

        # ── FFmpeg 拆帧为 30fps PNG ──
        max_w = self.RES_LIMITS.get(max_resolution, 1920)
        output_pattern = os.path.join(frames_dir, "frame_%05d.png")

        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-r", str(output_fps),
            "-vf", f"scale='min({max_w},iw)':-2:flags=lanczos",
            "-pix_fmt", "rgb24",
            "-compression_level", "0",     # PNG 最快压缩
            output_pattern
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 拆帧失败: {result.stderr}")

        # 统计实际输出帧数
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ])
        frame_count = len(frame_files)

        # ── 保存元数据 JSON ──
        metadata_path = os.path.join(frames_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2)

        # ── 写入 context.json ──
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)

        context.update({
            "input_video": video_path,
            "frames_dir": frames_dir,
            "audio_path": audio_path,
            "video_fps": output_fps,
            "original_fps": meta["fps"],
            "total_frames": frame_count,
            "resolution": f"{meta['width']}x{meta['height']}",
            "original_width": meta["width"],
            "original_height": meta["height"],
            "pix_fmt": meta["pix_fmt"],
            "color_space": meta["color_space"],
            "color_transfer": meta["color_transfer"],
            "color_primaries": meta["color_primaries"],
            "codec": meta["codec"],
            "bit_rate": meta["bit_rate"],
            "audio": meta.get("audio"),
        })

        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)

        logger.info(
            f"[MementoPreprocess] 完成: {frame_count} 帧输出到 {frames_dir}, "
            f"音频: {audio_path or '无'}"
        )
        return (frames_dir, audio_path, frame_count, metadata_path)


NODE_CLASS_MAPPINGS = {"MementoPreprocess": MementoPreprocess}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPreprocess": "Memento 01 - FFmpeg 预处理"}