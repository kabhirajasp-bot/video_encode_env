# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scalar reward for encoding steps:

    R = λ_Q · Q̄ − λ_T · T̄ − λ_S · S̄ − λ_C · C̄

- Q̄: quality in [0, 1] from VMAF (÷ VMAF_MAX) or SSIM when VMAF is missing
- T̄: encode time in [0, 1] via min(time / TIME_MAX, 1)
- S̄: bitrate cost in [0, 1] via min(bitrate_kbps / BITRATE_MAX, 1)
- C̄: CRF instability vs previous segment mean ``crf_avg``: 0 if
  ``|crf − crf_avg| / crf_avg ≤`` threshold; otherwise ramps in [0, 1] with excess
  relative deviation (no penalty before the first completed segment).
"""

from __future__ import annotations

from typing import Any


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def normalized_quality(
    vmaf: float | None,
    ssim: float | None,
    *,
    vmaf_max: float = 100.0,
) -> float:
    """Map VMAF or SSIM to [0, 1]. Prefer VMAF when present."""
    if vmaf is not None and vmaf_max > 0:
        return _clip01(vmaf / vmaf_max)
    if ssim is not None:
        return _clip01(ssim)
    return 0.0


def normalized_encode_time(encoding_time_sec: float, *, time_max_sec: float) -> float:
    if time_max_sec <= 0:
        return 0.0
    return _clip01(encoding_time_sec / time_max_sec)


def normalized_bitrate(bitrate_kbps: float, *, bitrate_max_kbps: float) -> float:
    """Bitrate cost term S̄ (higher bitrate → higher penalty weight)."""
    if bitrate_max_kbps <= 0:
        return 0.0
    return _clip01(bitrate_kbps / bitrate_max_kbps)


def normalized_crf_instability(
    crf: int,
    prev_crf_avg: float | None,
    *,
    relative_threshold: float = 0.5,
) -> float:
    """
    Penalty term C̄ in [0, 1] when current CRF differs from the previous segment's
    mean CRF by more than ``relative_threshold`` (relative, e.g. 0.5 = 50%).
    """
    if prev_crf_avg is None or prev_crf_avg <= 0:
        return 0.0
    rel_dev = abs(float(crf) - prev_crf_avg) / prev_crf_avg
    if rel_dev <= relative_threshold:
        return 0.0
    # map excess in (0, ...] to (0, 1]; cap at 1
    excess = rel_dev - relative_threshold
    return min(1.0, excess / max(1e-9, 1.0 - relative_threshold))


def compute_segment_reward(
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
