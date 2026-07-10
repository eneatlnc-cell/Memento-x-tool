"""
Memento 09 — FFmpeg Final Composite Node
=========================================
Combines the final composited frame sequence with the original audio
track into a single 4K video file using FFmpeg.

Handles the case where audio is missing by producing a silent video.
"""

import os
import subprocess
import glob


class MementoComposite:
    """ComfyUI custom node: Memento 09 — FFmpeg Final Composite.

    Uses FFmpeg to encode the frame sequence from 08_fusion together
    with the extracted audio from 01 into a high-quality .mp4 file.

    Inputs:
        final_frames_dir : path to directory of 08 final composite frames
        audio_path       : path to the separated original audio file
        original_fps     : frame rate from source metadata
        original_width   : output width in pixels
        original_height  : output height in pixels

    Outputs:
        output_video_path : path to the completed .mp4 file
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_frames_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to 08 final composite frames"
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to 01 separated original audio file"
                }),
                "original_fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.01,
                    "display": "number",
                }),
                "original_width": ("INT", {
                    "default": 3840,
                    "min": 1,
                    "max": 7680,
                    "step": 1,
                    "display": "number",
                }),
                "original_height": ("INT", {
                    "default": 2160,
                    "min": 1,
                    "max": 4320,
                    "step": 1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "composite"
    CATEGORY = "Memento/09_Composite"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg():
        """Locate the FFmpeg binary.  Returns the path or raises."""
        candidates = [
            "ffmpeg",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        for c in candidates:
            try:
                subprocess.run(
                    [c, "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                return c
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
        raise RuntimeError(
            "FFmpeg not found.  Please install FFmpeg and ensure it is on PATH."
        )

    @staticmethod
    def _detect_frame_pattern(frames_dir):
        """Detect the naming pattern of frames in *frames_dir*.

        Returns (pattern_path, start_number) suitable for FFmpeg -i.
        """
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(frames_dir, ext)))
        if not files:
            raise ValueError(f"No image files found in {frames_dir}")

        files.sort()

        # Determine the pattern: look for a common numeric sequence
        first = os.path.splitext(os.path.basename(files[0]))[0]
        ext   = os.path.splitext(files[0])[1]

        # Try to extract the numeric part at the end
        digits = ""
        for ch in reversed(first):
            if ch.isdigit():
                digits = ch + digits
            else:
                break

        if not digits:
            # Fall back to a simple glob wildcard
            pattern = os.path.join(frames_dir, f"frame_%06d{ext}")
            start_number = 0
        else:
            prefix = first[: -len(digits)] if len(digits) < len(first) else "frame_"
            width = len(digits)
            pattern = os.path.join(frames_dir, f"{prefix}%0{width}d{ext}")
            start_number = int(digits)

        return pattern, start_number

    @staticmethod
    def _has_audio(audio_path):
        """Return True if *audio_path* points to a readable audio file."""
        if not audio_path or not os.path.isfile(audio_path):
            return False
        # Quick check: try to probe with ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-select_streams", "a:0",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0",
                    audio_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            return "audio" in result.stdout.lower()
        except Exception:
            # If ffprobe is unavailable, assume the file is valid audio
            return os.path.getsize(audio_path) > 0

    # ------------------------------------------------------------------
    # Main composite pipeline
    # ------------------------------------------------------------------

    def composite(
        self,
        final_frames_dir,
        audio_path,
        original_fps,
        original_width,
        original_height,
    ):
        """Encode the final video.

        Returns the absolute path to the output .mp4 file.
        """
        # --- Validate inputs ---
        if not final_frames_dir or not os.path.isdir(final_frames_dir):
            raise ValueError(
                f"final_frames_dir is not a valid directory: {final_frames_dir}"
            )

        # --- Determine frame pattern ---
        frame_pattern, start_number = self._detect_frame_pattern(final_frames_dir)
        print(f"[MementoComposite] Frame pattern: {frame_pattern}")
        print(f"[MementoComposite] Start number: {start_number}")

        # --- Locate ffmpeg ---
        ffmpeg = self._find_ffmpeg()
        print(f"[MementoComposite] Using FFmpeg: {ffmpeg}")

        # --- Determine output path ---
        base_out = os.path.dirname(os.path.normpath(final_frames_dir))
        output_video_path = os.path.join(base_out, "09_final_output.mp4")

        # --- Build FFmpeg command ---
        has_audio = self._has_audio(audio_path)
        if has_audio:
            print(f"[MementoComposite] Audio source: {audio_path}")
        else:
            print("[MementoComposite] No valid audio found — producing silent video")

        # Base command: video input
        cmd = [
            ffmpeg,
            "-y",  # overwrite output
            "-start_number", str(start_number),
            "-framerate", str(original_fps),
            "-i", frame_pattern,
        ]

        # Audio input (if present)
        if has_audio:
            cmd += ["-i", audio_path]

        # Scale filter if needed (maintain aspect ratio)
        vf_parts = []
        vf_parts.append(f"scale={original_width}:{original_height}:force_original_aspect_ratio=decrease")
        vf_parts.append(f"pad={original_width}:{original_height}:(ow-iw)/2:(oh-ih)/2")
        vf_filter = ",".join(vf_parts)

        # Codec settings
        cmd += [
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-vf", vf_filter,
        ]

        if has_audio:
            cmd += [
                "-c:a", "aac",
                "-b:a", "320k",
                "-shortest",
            ]
        else:
            # Generate a silent audio track so the file is widely compatible
            cmd += [
                "-f", "lavfi",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-c:a", "aac",
                "-b:a", "128k",
                "-shortest",
            ]

        cmd.append(output_video_path)

        print(f"[MementoComposite] FFmpeg command:")
        print(f"  {' '.join(cmd)}")

        # --- Run FFmpeg ---
        print("[MementoComposite] Encoding video — this may take a while ...")
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=7200,  # 2-hour timeout
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("FFmpeg encoding timed out after 2 hours.")

        if result.returncode != 0:
            print("[MementoComposite] FFmpeg stderr:")
            print(result.stderr[-3000:])  # last 3000 chars
            raise RuntimeError(
                f"FFmpeg exited with code {result.returncode}.  "
                f"See log above for details."
            )

        # --- Verify output ---
        if not os.path.isfile(output_video_path):
            raise RuntimeError(
                f"Output file was not created: {output_video_path}"
            )

        file_size_mb = os.path.getsize(output_video_path) / (1024 * 1024)

        # Probe duration
        duration_str = "unknown"
        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    output_video_path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                duration_sec = float(probe.stdout.strip())
                mins, secs = divmod(duration_sec, 60)
                hours, mins = divmod(mins, 60)
                duration_str = f"{int(hours)}h {int(mins)}m {secs:.1f}s"
        except Exception:
            pass

        print(f"[MementoComposite] Output: {output_video_path}")
        print(f"[MementoComposite] File size:  {file_size_mb:.1f} MB")
        print(f"[MementoComposite] Duration:   {duration_str}")

        return (output_video_path,)


# ------------------------------------------------------------------
# ComfyUI registration
# ------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "MementoComposite": MementoComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MementoComposite": "Memento 09 — Composite",
}