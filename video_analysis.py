# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lightweight ffprobe-based metrics for whole-video and segment clips.

Per-clip **luma / complexity** features (same definitions for whole file and segment extracts):

- **EY** — luma texture / spatial complexity: FFmpeg ``siti`` filter *Spatial Information* (SI),
  ITU-style edge energy on the luma plane.
- **h** — temporal complexity: ``siti`` *Temporal Information* (TI), frame-to-frame variation.
- **LY** — luma brightness: mean of per-frame ``signalstats`` ``YAVG`` (0–255).

Observation dict keys (use these names in code and for agents):

- ``luma_spatial_texture_complexity_EY`` — maps to **EY** / SI above
- ``temporal_complexity_h`` — maps to **h** / TI above
- ``mean_luma_brightness_LY`` — maps to **LY** / mean YAVG above
"""

from __future__ import annotations

import copy
import re
import subprocess
from pathlib import Path
from typing import Any

# Full JSON keys for complexity metrics (paper symbols EY, h, LY kept in the name for traceability).
LUMA_SPATIAL_TEXTURE_COMPLEXITY_EY = "luma_spatial_texture_complexity_EY"
TEMPORAL_COMPLEXITY_H = "temporal_complexity_h"
MEAN_LUMA_BRIGHTNESS_LY = "mean_luma_brightness_LY"

# Populated on ``POST /api/videos/upload``; read by the env on reset/step (no ffmpeg there).
# Process-local only; cleared on server restart.
_video_analysis_memory: dict[str, dict[str, Any]] = {}


def _video_path_key(path: Path) -> str:
    return str(path.resolve())


def store_whole_video_analysis(video_path: Path, data: dict[str, Any]) -> None:
    """Keep ``analyze_video`` output in memory for later observations (keyed by resolved path)."""
    _video_analysis_memory[_video_path_key(video_path)] = copy.deepcopy(data)


def forget_whole_video_analysis(video_path: Path) -> None:
    """Drop cached analysis for ``video_path`` (e.g. after the file is deleted)."""
    _video_analysis_memory.pop(_video_path_key(video_path), None)


def load_whole_video_analysis_for_observation(video_path: Path) -> dict[str, Any]:
    """
    Data for ``VideoEncodeObservation.whole_video_analysis`` without running ffmpeg.

    Returns a copy of the in-memory snapshot from upload if present and file size still matches;
    otherwise ``{}``.
    """
    key = _video_path_key(video_path)
    data = _video_analysis_memory.get(key)
    if data is None:
        return {}
    try:
        sz = video_path.stat().st_size
    except OSError:
        return {}
    fsb = data.get("file_size_bytes")
    if fsb is not None and int(fsb) != sz:
        return {}
    return copy.deepcopy(data)


def _run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _parse_frame_rate(s: str) -> float | None:
    s = s.strip()
    if not s or s == "0/0":
        return None
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            n, d = float(num), float(den)
            return n / d if d else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def ffprobe_duration_sec(path: Path) -> float:
    r = _run_ffprobe(
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
    r = _run_ffprobe(
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


def ffprobe_bitrate_kbps(path: Path) -> float:
    r = _run_ffprobe(
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


_YAVG_RE = re.compile(r"lavfi\.signalstats\.YAVG=([0-9.+-eE]+)")


def _parse_siti_spatial_temporal_averages(stderr: str) -> tuple[float | None, float | None]:
    """Return (SI average, TI average) from ``siti=print_summary=1`` stderr."""
    si: float | None = None
    ti: float | None = None
    section: str | None = None
    for line in stderr.splitlines():
        if "Spatial Information:" in line:
            section = "spatial"
        elif "Temporal Information:" in line:
            section = "temporal"
        elif line.strip().startswith("Average:"):
            try:
                val = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                continue
            if section == "spatial":
                si = val
                section = None
            elif section == "temporal":
                ti = val
                section = None
    return si, ti


def _mean_signalstats_yavg(stdout: str) -> float | None:
    vals = [float(m.group(1)) for m in _YAVG_RE.finditer(stdout)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _run_ffmpeg_luma_complexity_metrics(
    path: Path,
    *,
    complexity_timeout_sec: float,
) -> dict[str, float | None]:
    """Compute EY (SI), h (TI), LY (mean YAVG) via two ffmpeg passes."""
    luma_spatial_texture_complexity_EY: float | None = None
    temporal_complexity_h: float | None = None
    mean_luma_brightness_LY: float | None = None

    r_siti = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "siti=print_summary=1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=complexity_timeout_sec,
        check=False,
    )
    si, ti = _parse_siti_spatial_temporal_averages(r_siti.stderr)
    luma_spatial_texture_complexity_EY = si
    temporal_complexity_h = ti

    r_stat = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:file=-",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=complexity_timeout_sec,
        check=False,
    )
    if r_stat.returncode == 0:
        mean_luma_brightness_LY = _mean_signalstats_yavg(r_stat.stdout)

    return {
        LUMA_SPATIAL_TEXTURE_COMPLEXITY_EY: luma_spatial_texture_complexity_EY,
        TEMPORAL_COMPLEXITY_H: temporal_complexity_h,
        MEAN_LUMA_BRIGHTNESS_LY: mean_luma_brightness_LY,
    }


def ffprobe_avg_frame_rate(path: Path) -> float | None:
    r = _run_ffprobe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    if r.returncode != 0:
        return None
    line = r.stdout.strip()
    if not line:
        return None
    for p in line.split(","):
        fr = _parse_frame_rate(p)
        if fr is not None and fr > 0:
            return fr
    return None


def analyze_video(path: Path, *, complexity_timeout_sec: float = 120.0) -> dict[str, Any]:
    """Whole-file metrics (duration, resolution, fps, bitrate, size, luma/complexity keys)."""
    out: dict[str, Any] = {"path": str(path.resolve())}
    try:
        out["duration_sec"] = ffprobe_duration_sec(path)
    except RuntimeError as e:
        out["duration_sec"] = None
        out["error"] = str(e)
        return out
    try:
        w, h = ffprobe_video_size(path)
        out["width"] = w
        out["height"] = h
    except RuntimeError:
        out["width"] = None
        out["height"] = None
    try:
        out["bitrate_kbps"] = ffprobe_bitrate_kbps(path)
    except RuntimeError:
        out["bitrate_kbps"] = None
    out["fps"] = ffprobe_avg_frame_rate(path)
    try:
        out["file_size_bytes"] = path.stat().st_size
    except OSError:
        out["file_size_bytes"] = None

    try:
        metrics = _run_ffmpeg_luma_complexity_metrics(
            path,
            complexity_timeout_sec=complexity_timeout_sec,
        )
        out[LUMA_SPATIAL_TEXTURE_COMPLEXITY_EY] = metrics[LUMA_SPATIAL_TEXTURE_COMPLEXITY_EY]
        out[TEMPORAL_COMPLEXITY_H] = metrics[TEMPORAL_COMPLEXITY_H]
        out[MEAN_LUMA_BRIGHTNESS_LY] = metrics[MEAN_LUMA_BRIGHTNESS_LY]
    except (TimeoutError, subprocess.TimeoutExpired, OSError):
        out[LUMA_SPATIAL_TEXTURE_COMPLEXITY_EY] = None
        out[TEMPORAL_COMPLEXITY_H] = None
        out[MEAN_LUMA_BRIGHTNESS_LY] = None

    return out
