# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-step grader functions for the Video Encode RL environment.

Each returns a score in [0, 1] based on a single step's measurable outcomes.
These are verifiable, Markovian signals suitable for use as shaped training rewards.

  easy_step_score   — structural validity: encode success + minimum watchable quality
  medium_step_score — rate-distortion efficiency: quality-per-bit tradeoff
"""

from __future__ import annotations


def easy_step_score(
    vmaf: float | None,
    encoding_time_sec: float | None,
    segment_duration_sec: float,
    encode_aborted: bool,
    *,
    vmaf_floor: float = 45.0,
    time_budget_multiplier: float = 2.0,
) -> float:
    """
    Score in [0, 1] for structural validity of one encode step.

    Deductions (result clipped to [0, 1]):
      encode_aborted or vmaf is None  →  −0.5  (encode did not complete)
      vmaf < vmaf_floor               →  −0.3  (video is unwatchable)
      enc_time > multiplier × seg_len →  −0.2  (encode far too slow)
    """
    score = 1.0
    if encode_aborted or vmaf is None:
        score -= 0.5
    elif vmaf < vmaf_floor:
        score -= 0.3
    if (
        encoding_time_sec is not None
        and segment_duration_sec > 0
        and encoding_time_sec > time_budget_multiplier * segment_duration_sec
    ):
        score -= 0.2
    return max(0.0, min(1.0, score))


def medium_step_score(
    vmaf: float | None,
    bitrate_kbps: float | None,
    *,
    bitrate_cap_kbps: float = 2000.0,
    vmaf_max: float = 100.0,
) -> float:
    """
    Score in [0, 1] for rate-distortion efficiency of one encode step.

        R_medium = (vmaf / vmaf_max) × (1 − clip(bitrate / bitrate_cap, 0, 1))

    High score requires simultaneously high quality AND low bitrate.
    Returns 0.0 when vmaf or bitrate is unavailable.
    """
    if vmaf is None or bitrate_kbps is None:
        return 0.0
    q = max(0.0, min(1.0, vmaf / vmaf_max))
    s = max(0.0, min(1.0, bitrate_kbps / bitrate_cap_kbps))
    return q * (1.0 - s)
