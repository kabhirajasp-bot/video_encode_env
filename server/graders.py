# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-step grader functions for the Video Encode RL environment.

Each returns a score in [0, 1] based on a single step's measurable outcomes.
These are verifiable, Markovian signals suitable for use as shaped training rewards.

  grader_easy   — structural validity: encode success + minimum watchable quality
  grader_medium — rate-distortion efficiency: quality-per-bit tradeoff
  grader_hard   — full quality-time-bitrate reward (R = λ_Q·Q̄ − λ_T·T̄ − λ_S·S̄ − λ_C·C̄)
"""

from __future__ import annotations

from typing import Any


def grader_easy(
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


def grader_medium(
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


def grader_hard(
    *,
    vmaf: float | None,
    ssim: float | None,
    encoding_time_sec: float,
    bitrate_kbps: float,
    lambda_q: float,
    lambda_t: float,
    lambda_s: float,
    crf: int = 0,
    prev_crf_avg: float | None = None,
    lambda_c: float = 0.0,
    crf_instability_relative_threshold: float = 0.5,
    vmaf_max: float = 100.0,
    time_max_sec: float = 60.0,
    bitrate_max_kbps: float = 20_000.0,
) -> tuple[float, dict[str, Any]]:
    """
    Return (reward, components) where components holds normalized terms for logging.

    R = λ_Q · Q̄ − λ_T · T̄ − λ_S · S̄ − λ_C · C̄
    """
    try:
        from .reward_utils import (
            normalized_bitrate,
            normalized_crf_instability,
            normalized_encode_time,
            normalized_quality,
        )
    except ImportError:
        from server.reward_utils import (
            normalized_bitrate,
            normalized_crf_instability,
            normalized_encode_time,
            normalized_quality,
        )

    q_bar = normalized_quality(vmaf, ssim, vmaf_max=vmaf_max)
    t_bar = normalized_encode_time(encoding_time_sec, time_max_sec=time_max_sec)
    s_bar = normalized_bitrate(bitrate_kbps, bitrate_max_kbps=bitrate_max_kbps)
    c_bar = normalized_crf_instability(
        crf,
        prev_crf_avg,
        relative_threshold=crf_instability_relative_threshold,
    )

    reward = (
        lambda_q * q_bar
        - lambda_t * t_bar
        - lambda_s * s_bar
        - lambda_c * c_bar
    )

    return reward, {
        "q_bar": q_bar,
        "t_bar": t_bar,
        "s_bar": s_bar,
        "c_bar": c_bar,
        "lambda_q": lambda_q,
        "lambda_t": lambda_t,
        "lambda_s": lambda_s,
        "lambda_c": lambda_c,
        "term_q": lambda_q * q_bar,
        "term_t": -lambda_t * t_bar,
        "term_s": -lambda_s * s_bar,
        "term_c": -lambda_c * c_bar,
        "prev_crf_avg": prev_crf_avg,
        "crf_instability_relative_threshold": crf_instability_relative_threshold,
    }
