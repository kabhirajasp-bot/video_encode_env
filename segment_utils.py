# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FFmpeg helpers: probe, segment extraction, encode, VMAF/SSIM, bitrate."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _run_ffmpeg(args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def ffprobe_duration_sec(path: Path) -> float:
    """Return container duration in seconds."""
    r = _run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr}")
    return float(r.stdout.strip())


def ffprobe_video_size(path: Path) -> tuple[int, int]:
    """Return width, height of the first video stream."""
    r = _run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe size failed for {path}: {r.stderr}")
    parts = r.stdout.strip().split("x")
    if len(parts) != 2:
        raise RuntimeError(f"unexpected ffprobe size output: {r.stdout!r}")
    return int(parts[0]), int(parts[1])


def extract_segment(
    input_path: Path,
    start_sec: float,
    duration_sec: float,
    output_path: Path,
    *,
    timeout: float | None = None,
) -> None:
    """Extract [start_sec, start_sec+duration) to output_path (re-encode for robust cuts)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.4f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration_sec:.4f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    r = _run_ffmpeg(args, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extract failed: {r.stderr}")


def encode_segment(
    input_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    crf: int,
    preset: str,
    timeout: float | None = None,
) -> float:
    """Encode input to H.264 with scale + CRF. Returns wall-clock seconds for the encode."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={width}:{height}:flags=bicubic"
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    t0 = time.perf_counter()
    r = _run_ffmpeg(args, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {r.stderr}")
    return elapsed


def ffprobe_bitrate_kbps(path: Path) -> float:
    """Average bitrate in kbps from container format."""
    r = _run_ffmpeg(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=bit_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe bitrate failed: {r.stderr}")
    line = r.stdout.strip()
    if not line or line == "N/A":
        size = path.stat().st_size
        dur = ffprobe_duration_sec(path)
        if dur <= 0:
            return 0.0
        return (size * 8) / dur / 1000.0
    return float(line) / 1000.0


def _read_vmaf_json(log_path: Path) -> float | None:
    try:
        data = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    pm = data.get("pooled_metrics")
    if isinstance(pm, dict) and "vmaf" in pm:
        v = pm["vmaf"]
        if isinstance(v, dict) and "mean" in v:
            return float(v["mean"])
    if "vmaf" in data and isinstance(data["vmaf"], dict):
        m = data["vmaf"].get("mean")
        if m is not None:
            return float(m)
    frames = data.get("frames")
    if isinstance(frames, list) and frames:
        metrics = frames[-1].get("metrics") or {}
        vmaf = metrics.get("vmaf")
        if vmaf is not None:
            return float(vmaf)
    return None


def _vmaf_from_stderr(stderr: str) -> float | None:
    m = re.search(r"VMAF[\s\w]*[:=]\s*([0-9]+(?:\.[0-9]+)?)", stderr, re.I)
    if m:
        return float(m.group(1))
    return None


def vmaf_score(
    distorted_path: Path,
    reference_path: Path,
    *,
    timeout: float | None = None,
) -> float | None:
    """
    Pooled VMAF mean (0..100) via libvmaf.
    Input 0 = distorted, input 1 = reference. Both streams are scaled to the
    distorted file's width/height — libvmaf requires matching dimensions on both inputs.
    """
    try:
        w, h = ffprobe_video_size(distorted_path)
    except RuntimeError as e:
        logger.warning("vmaf: could not read distorted size: %s", e)
        return None

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        log_path = Path(tmp.name)
    try:
        lp = str(log_path)
        fc = (
            f"[0:v]scale={w}:{h}:flags=bicubic[dist];"
            f"[1:v]scale={w}:{h}:flags=bicubic[ref];"
            f"[dist][ref]libvmaf=log_path={lp}:log_fmt=json"
        )
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(distorted_path),
            "-i",
            str(reference_path),
            "-filter_complex",
            fc,
            "-f",
            "null",
            "-",
        ]
        r = _run_ffmpeg(args, timeout=timeout)
        pooled = _read_vmaf_json(log_path)
        if pooled is not None:
            return pooled
        if r.returncode != 0:
            logger.warning("libvmaf failed: %s", r.stderr)
        return _vmaf_from_stderr(r.stderr or "")
    finally:
        if log_path.exists():
            log_path.unlink(missing_ok=True)


def ssim_score(
    distorted_path: Path,
    reference_path: Path,
    *,
    timeout: float | None = None,
) -> float | None:
    """
    SSIM All (roughly 0..1). Distorted first, reference second.
    Both streams are scaled to the distorted resolution (same as VMAF).
    """
    try:
        w, h = ffprobe_video_size(distorted_path)
    except RuntimeError as e:
        logger.warning("ssim: could not read distorted size: %s", e)
        return None

    fc = (
        f"[0:v]scale={w}:{h}:flags=bicubic[dist];"
        f"[1:v]scale={w}:{h}:flags=bicubic[ref];"
        f"[dist][ref]ssim"
    )
    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(distorted_path),
        "-i",
        str(reference_path),
        "-filter_complex",
        fc,
        "-f",
        "null",
        "-",
    ]
    r = _run_ffmpeg(args, timeout=timeout)
    if r.returncode != 0:
        logger.warning("ssim filter failed: %s", r.stderr)
        return None
    text = r.stderr + r.stdout
    m = re.search(r"All:([0-9.]+)", text)
    if m:
        return float(m.group(1))
    return None


def build_segment_features(
    segment_path: Path,
    *,
    video_id: str,
    segment_index: int,
) -> dict[str, Any]:
    w, h = ffprobe_video_size(segment_path)
    dur = ffprobe_duration_sec(segment_path)
    return {
        "segment_id": f"{video_id}_{segment_index:05d}",
        "video_id": video_id,
        "segment_index": segment_index,
        "duration_sec": dur,
        "width": w,
        "height": h,
        "file_size_bytes": segment_path.stat().st_size,
    }
