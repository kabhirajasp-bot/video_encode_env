# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Video Encode Environment.

Observations include segment index, DQ-style previous-segment predictions,
and ffprobe-based whole-video analysis dicts.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class VideoEncodeAction(Action):
    """Encoding parameters for one step (output size follows reference segment resolution)."""

    crf: int = Field(23, ge=0, le=51, description="libx264 CRF")
    preset: str = Field("medium", description="libx264 preset, e.g. fast, medium, slow")
    video_index: Optional[int] = Field(
        None,
        description="Index into the init video list; default uses current episode video from reset",
    )


class VideoEncodeObservation(Observation):
    """Encode metrics plus segment state and analysis features."""

    echoed_message: str = Field(default="", description="Status or error summary")
    num_videos: int = Field(default=0, description="Number of videos loaded at init")
    current_video_index: int = Field(default=0, description="Active video index after last reset")
    reset_index: int = Field(default=0, description="Increments each reset")
    segment_index: int = Field(
        default=0,
        description="Timeline segment index (0-based along the file) for this encode step",
    )
    encode_step_index: int = Field(
        default=0,
        description="Which sub-step within the segment (0 .. steps_per_segment-1) this step was",
    )
    steps_per_segment: int = Field(
        default=10,
        description="K: encode attempts per timeline segment before advancing",
    )
    prev_segment_predictions: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Summary of the last completed timeline segment: means over K steps "
            "(vmaf/ssim/bitrate/time/reward, crf_avg, presets_used); None before first advance"
        ),
    )
    whole_video_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metrics for the full source file (duration, resolution, …) plus "
            "luma_spatial_texture_complexity_EY, temporal_complexity_h, mean_luma_brightness_LY "
            "(see video_analysis module constants). Filled from in-memory analysis cached at "
            "upload time; empty dict if the file was not analyzed in this process."
        ),
    )
    segment_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-segment features computed on the extracted reference clip for this step. "
            "Always-present keys (may be None on probe failure): width, height, duration_sec, "
            "fps, bitrate_kbps, file_size_bytes. "
            "Complexity keys (when analysis succeeds): "
            "luma_spatial_texture_complexity_EY (SI), temporal_complexity_h (TI), "
            "mean_luma_brightness_LY. "
            "Empty dict when no clip was extracted this step (e.g. reset, end-of-video, "
            "or pre-extraction error)."
        ),
    )
    vmaf_score: Optional[float] = Field(default=None, description="VMAF pooled mean vs reference segment")
    ssim_score: Optional[float] = Field(default=None, description="SSIM All if VMAF unavailable")
    bitrate_kbps: Optional[float] = Field(default=None, description="Average bitrate of encoded output")
    encoding_time_sec: Optional[float] = Field(default=None, description="Wall time for encode only")
