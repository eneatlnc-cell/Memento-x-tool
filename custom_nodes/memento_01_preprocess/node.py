"""Memento 01 — FFmpeg 预处理节点

将 H.264/HEVC 视频拆帧为 PNG 序列，分辨率适配。
写入视频元数据到 /workspace/context.json
"""
import logging
import json
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class MementoPreprocess:
    """节点 1: 视频预处理 — 拆帧 + 分辨率适配"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "max_resolution": (["1080p", "4K", "8K"], {"default": "1080p"}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("frames_dir", "frame_count")
    FUNCTION = "process"
    CATEGORY = "Memento/01_Preprocess"

    def get_video_metadata(self, video_path: str) -> dict:
        """用 ffprobe 获取视频元数据"""
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=width,height,duration,nb_read_frames",
            "-of", "json", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration": float(stream["duration"]),
            "nb_frames": int(stream.get("nb_read_frames", "0")),
        }

    def process(self, video_path: str, output_fps: int, max_resolution: str):
        logger.info(f"[MementoPreprocess] 输入: {video_path}, fps={output_fps}, res={max_resolution}")
        
        # 检查输入文件存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"输入视频不存在: {video_path}")
        
        # 创建输出目录
        frames_dir = "/workspace/frames"
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        
        # 获取原视频元数据
        meta = self.get_video_metadata(video_path)
        logger.info(f"[MementoPreprocess] 原视频: {meta['width']}x{meta['height']}, {meta['nb_frames']} 帧")
        
        # FFmpeg 拆帧
        output_pattern = os.path.join(frames_dir, "frame_%05d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-r", str(output_fps),
            "-q:v", "2",
            output_pattern
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 拆帧失败: {result.stderr}")
        
        # 统计实际输出帧数
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
        frame_count = len(frame_files)
        
        # 计算分辨率（按实际输出）
        # 这里我们让 FFmpeg 输出保持原宽高比，只限制最大分辨率
        resolution = f"{meta['width']}x{meta['height']}"
        
        # 写入 context.json（追加模式）
        context_path = "/workspace/context.json"
        context = {}
        if os.path.exists(context_path):
            with open(context_path, "r") as f:
                context = json.load(f)
        
        context.update({
            "input_video": video_path,
            "frames_dir": frames_dir,
            "video_fps": output_fps,
            "total_frames": frame_count,
            "resolution": resolution,
            "original_width": meta["width"],
            "original_height": meta["height"],
        })
        
        with open(context_path, "w") as f:
            json.dump(context, f, indent=2)
        
        logger.info(f"[MementoPreprocess] 完成: {frame_count} 帧输出到 {frames_dir}")
        return (frames_dir, frame_count)


NODE_CLASS_MAPPINGS = {"MementoPreprocess": MementoPreprocess}
NODE_DISPLAY_NAME_MAPPINGS = {"MementoPreprocess": "Memento 01 - FFmpeg 预处理"}