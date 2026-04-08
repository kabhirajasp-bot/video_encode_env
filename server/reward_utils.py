# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Normalisation helpers used by grader_hard (server/graders.py).

    Q̄: quality in [0, 1] from VMAF (÷ VMAF_MAX) or SSIM when VMAF is missing
    T̄: encode time in [0, 1] via min(time / TIME_MAX, 1)
    S̄: bitrate cost in [0, 1] via min(bitrate_kbps / BITRATE_MAX, 1)
    C̄: CRF instability vs previous segment mean crf_avg
"""

from __future__ import annotations


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
