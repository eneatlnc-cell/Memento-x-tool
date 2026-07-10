"""
FFmpeg Streaming Video Decoder

A complete decoder that reads video frames via FFmpeg subprocess piping,
producing torch.Tensor chunks in RGB order with float32 [0,1] normalization.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ffprobe(video_path: str) -> Dict[str, Any]:
    """Run ffprobe on *video_path* and return the parsed JSON output."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffprobe exited with code {proc.returncode}: {proc.stderr.strip()}"
            )
        return json.loads(proc.stdout)
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found in PATH.  Please install FFmpeg."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe timed out after 60 s.")


def _safe_float(value: Any) -> Optional[float]:
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert *value* to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# StreamDecoder
# ---------------------------------------------------------------------------

class StreamDecoder:
    """Decode a video file into torch tensors by piping FFmpeg raw RGB24 output.

    Parameters
    ----------
    video_path : str
        Absolute or relative path to the source video file.
    """

    def __init__(self, video_path: str) -> None:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Check file size proactively (1 GB limit)
        file_size = os.path.getsize(video_path)
        if file_size > 1 * 1024 * 1024 * 1024:
            raise ValueError(
                f"Video file exceeds 1 GB limit ({file_size / (1024**3):.2f} GB)."
            )

        self.video_path = video_path
        self._metadata: Optional[Dict[str, Any]] = None
        self._width: Optional[int] = None
        self._height: Optional[int] = None
        self._frame_bytes: Optional[int] = None
        self._fps: Optional[float] = None
        self._duration: Optional[float] = None
        self._nb_frames: Optional[int] = None
        self._pix_fmt: Optional[str] = None
        self._color_space: Optional[str] = None
        self._color_transfer: Optional[str] = None
        self._color_primaries: Optional[str] = None
        self._codec: Optional[str] = None
        self._bit_rate: Optional[int] = None
        self._audio: Dict[str, Any] = {}

        logger.info("StreamDecoder initialized for %s (%.2f MB)",
                     video_path, file_size / (1024**2))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Any]:
        """Return a dict with video and (optional) audio metadata.

        Keys
        ----
        width, height, fps, duration, nb_frames, pix_fmt,
        color_space, color_transfer, color_primaries, codec, bit_rate,
        audio : dict with keys ``codec``, ``sample_rate``, ``channels``
            (empty dict when no audio stream is present).
        """
        if self._metadata is not None:
            return self._metadata

        raw = _ffprobe(self.video_path)

        # Find the first video stream
        video_stream = None
        audio_stream = None
        for stream in raw.get("streams", []):
            codec_type = stream.get("codec_type", "")
            if codec_type == "video" and video_stream is None:
                video_stream = stream
            elif codec_type == "audio" and audio_stream is None:
                audio_stream = stream

        if video_stream is None:
            raise RuntimeError("No video stream found in the file.")

        # --- video fields -------------------------------------------------
        self._width = _safe_int(video_stream.get("width"))
        self._height = _safe_int(video_stream.get("height"))
        if self._width is None or self._height is None:
            raise RuntimeError("Could not determine video dimensions.")

        # frame byte size: 3 bytes per pixel (RGB24)
        self._frame_bytes = self._width * self._height * 3

        # fps
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = fps_str.split("/")
            self._fps = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            self._fps = _safe_float(video_stream.get("avg_frame_rate"))
            if self._fps is None:
                self._fps = 0.0

        # duration
        self._duration = _safe_float(video_stream.get("duration"))
        if self._duration is None:
            self._duration = _safe_float(raw.get("format", {}).get("duration", 0.0))

        # nb_frames
        self._nb_frames = _safe_int(video_stream.get("nb_frames"))
        if self._nb_frames is None or self._nb_frames == 0:
            # Estimate from duration * fps
            if self._duration and self._fps:
                self._nb_frames = int(round(self._duration * self._fps))
            else:
                self._nb_frames = 0

        # pixel format / colour metadata
        self._pix_fmt = video_stream.get("pix_fmt", "unknown")
        self._color_space = video_stream.get("color_space", "unknown")
        self._color_transfer = video_stream.get("color_transfer", "unknown")
        self._color_primaries = video_stream.get("color_primaries", "unknown")
        self._codec = video_stream.get("codec_name", "unknown")

        # bit rate (prefer stream-level, fall back to container-level)
        self._bit_rate = _safe_int(video_stream.get("bit_rate"))
        if self._bit_rate is None:
            self._bit_rate = _safe_int(raw.get("format", {}).get("bit_rate"))

        # --- audio fields -------------------------------------------------
        if audio_stream is not None:
            self._audio = {
                "codec": audio_stream.get("codec_name", "unknown"),
                "sample_rate": _safe_int(audio_stream.get("sample_rate")) or 0,
                "channels": _safe_int(audio_stream.get("channels")) or 0,
            }
        else:
            self._audio = {}

        self._metadata = {
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "duration": self._duration,
            "nb_frames": self._nb_frames,
            "pix_fmt": self._pix_fmt,
            "color_space": self._color_space,
            "color_transfer": self._color_transfer,
            "color_primaries": self._color_primaries,
            "codec": self._codec,
            "bit_rate": self._bit_rate,
            "audio": self._audio,
        }

        logger.info("Metadata: %dx%d @ %.2f fps, %d frames, codec=%s",
                     self._width, self._height, self._fps,
                     self._nb_frames, self._codec)
        return self._metadata

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(self, output_dir: str) -> str:
        """Extract the first audio stream as a 16-bit PCM WAV (48 kHz, stereo).

        Parameters
        ----------
        output_dir : str
            Directory where ``<basename>.wav`` will be written.

        Returns
        -------
        str
            Absolute path to the extracted WAV file, or an empty string
            if the video has no audio stream.
        """
        meta = self.get_metadata()
        if not meta.get("audio"):
            logger.info("No audio stream present – skipping extraction.")
            return ""

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.video_path))[0]
        out_path = os.path.join(output_dir, f"{base}.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-i", self.video_path,
            "-vn",              # drop video
            "-acodec", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            out_path,
        ]
        logger.info("Extracting audio: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                logger.error("Audio extraction failed: %s", proc.stderr.strip())
                return ""
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found in PATH.")
        except subprocess.TimeoutExpired:
            logger.error("Audio extraction timed out.")
            return ""

        if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
            logger.info("Audio saved to %s", out_path)
            return out_path
        else:
            logger.warning("Audio extraction produced no output file.")
            return ""

    # ------------------------------------------------------------------
    # Frame decoding
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        if self._width is None:
            self.get_metadata()
        return self._width  # type: ignore[return-value]

    @property
    def height(self) -> int:
        if self._height is None:
            self.get_metadata()
        return self._height  # type: ignore[return-value]

    @property
    def frame_bytes(self) -> int:
        if self._frame_bytes is None:
            self.get_metadata()
        return self._frame_bytes  # type: ignore[return-value]

    @property
    def nb_frames(self) -> int:
        if self._nb_frames is None:
            self.get_metadata()
        return self._nb_frames  # type: ignore[return-value]

    def _build_ffmpeg_cmd(self) -> list:
        """Build the ffmpeg subprocess command for raw RGB24 pipe."""
        return [
            "ffmpeg",
            "-v", "error",
            "-i", self.video_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-an",              # no audio in the pipe
            "-",
        ]

    def decode_chunks(
        self, chunk_size: int = 30
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """Decode video frames in chunks via a FFmpeg subprocess pipe.

        Yields
        ------
        (chunk_idx, frames_tensor)
            chunk_idx : int, zero-based chunk index.
            frames_tensor : torch.Tensor of shape ``(N, 3, H, W)``,
            dtype float32, values in [0, 1].

        The last chunk may be smaller than *chunk_size* when the total
        frame count is not evenly divisible.
        """
        self.get_metadata()  # ensure width/height/frame_bytes are known
        frame_byte_size = self.frame_bytes  # w * h * 3
        w, h = self.width, self.height

        cmd = self._build_ffmpeg_cmd()
        logger.debug("Starting ffmpeg pipe: %s", " ".join(cmd))

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * 1024 * 1024,  # 10 MB buffer
            )

            chunk_idx = 0
            frames_in_chunk: list[np.ndarray] = []

            while True:
                raw = proc.stdout.read(frame_byte_size)  # type: ignore[union-attr]
                if not raw:
                    break

                if len(raw) != frame_byte_size:
                    logger.warning(
                        "Partial frame read: expected %d bytes, got %d. Skipping.",
                        frame_byte_size, len(raw),
                    )
                    # Drain the rest and break
                    continue

                # Convert raw bytes -> numpy uint8 -> HWC -> CHW -> float32 [0,1]
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                frames_in_chunk.append(frame_np)

                if len(frames_in_chunk) >= chunk_size:
                    yield self._frames_to_tensor(frames_in_chunk, chunk_idx)
                    chunk_idx += 1
                    frames_in_chunk.clear()

            # Yield any remaining frames
            if frames_in_chunk:
                yield self._frames_to_tensor(frames_in_chunk, chunk_idx)

            # Check for ffmpeg errors
            retcode = proc.wait()
            if retcode != 0:
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace").strip()  # type: ignore[union-attr]
                logger.error(
                    "ffmpeg exited with code %d: %s", retcode, stderr_output
                )
                raise RuntimeError(
                    f"ffmpeg exited with code {retcode}: {stderr_output}"
                )

        except GeneratorExit:
            # Generator was closed externally – clean up
            logger.debug("decode_chunks generator closed; terminating ffmpeg.")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        except Exception:
            logger.exception("Error during decode_chunks.")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise
        finally:
            # Safety net: ensure process is cleaned up
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()

    def _frames_to_tensor(
        self, frames: list[np.ndarray], chunk_idx: int
    ) -> Tuple[int, torch.Tensor]:
        """Convert a list of HWC uint8 numpy frames into a batched torch tensor.

        Input:  list of np.ndarray, each shape (H, W, 3) dtype uint8
        Output: tensor of shape (N, 3, H, W) dtype float32 in [0, 1]
        """
        # Stack into a single numpy array: (N, H, W, 3)
        batch_np = np.stack(frames, axis=0)
        # Reorder to (N, 3, H, W) and convert to float32 [0, 1]
        batch_np = batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch_np)
        logger.debug("Chunk %d: shape %s, dtype %s", chunk_idx, tensor.shape, tensor.dtype)
        return chunk_idx, tensor

    def decode_all(self) -> torch.Tensor:
        """Decode the entire video and return a single tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(N, 3, H, W)``, dtype float32, values in [0, 1].
        """
        all_frames: list[np.ndarray] = []
        w, h = self.width, self.height
        frame_byte_size = self.frame_bytes

        cmd = self._build_ffmpeg_cmd()
        logger.debug("decode_all: starting ffmpeg pipe: %s", " ".join(cmd))

        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 * 1024 * 1024,
            )

            while True:
                raw = proc.stdout.read(frame_byte_size)  # type: ignore[union-attr]
                if not raw:
                    break
                if len(raw) != frame_byte_size:
                    logger.warning(
                        "Partial frame read: expected %d bytes, got %d. Skipping.",
                        frame_byte_size, len(raw),
                    )
                    continue
                frame_np = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
                all_frames.append(frame_np)

            retcode = proc.wait()
            if retcode != 0:
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace").strip()  # type: ignore[union-attr]
                raise RuntimeError(
                    f"ffmpeg exited with code {retcode}: {stderr_output}"
                )

        except Exception:
            logger.exception("Error during decode_all.")
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            raise

        if not all_frames:
            logger.warning("decode_all: no frames decoded.")
            return torch.empty((0, 3, h, w), dtype=torch.float32)

        batch_np = np.stack(all_frames, axis=0)  # (N, H, W, 3)
        batch_np = batch_np.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch_np)
        logger.info("decode_all: decoded %d frames, tensor shape %s", len(all_frames), tensor.shape)
        return tensor