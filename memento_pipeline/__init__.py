"""Memento Pipeline — GPU 张量流全流程编排

使用方式:
  from memento_pipeline.runner import PipelineRunner, run_pipeline

  # 完整流程
  runner = PipelineRunner(
      video_path="/path/to/video.mp4",
      click_points=[(0.5, 0.5, 1)],
      reference_dir="/path/to/reference_images",
      prompt="角色B描述",
  )
  output_path = runner.run()

  # 一键运行
  output_path = run_pipeline("/path/to/video.mp4", click_points=[(0.5, 0.5, 1)])

架构:
  memento_pipeline/
    runner.py           ← 主编排器 (分片循环 + 显存管理)
    stream_decoder.py   ← FFmpeg 流式解码器
    ops/
      __init__.py       ← 02-05 GPU 张量操作 (SAM3/MediaPipe/MotionBERT/Align)
      sub.py            ← 06-09 GPU 张量操作 (LTX/RAFT/Fusion/Composite)
"""
from .runner import PipelineRunner, run_pipeline
from .stream_decoder import StreamDecoder

__all__ = ["PipelineRunner", "run_pipeline", "StreamDecoder"]