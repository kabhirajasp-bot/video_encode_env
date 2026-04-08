# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Video Encode Environment (v1).

- **Init:** loads only a list of video paths (directory scan or list file).
- **Reset:** picks the active video index; resets timeline cursor and sub-step counter.
- **Step:** up to **K** encodes on the same timeline clip; then advances the cursor and stores
  **prev_segment_predictions** (means over those **K** steps: VMAF/SSIM, bitrate, time, reward, **crf_avg**,
  **presets_used**).

Environment variables:

- ``VIDEO_ENCODE_VIDEOS_DIR`` — directory of ``.mp4/.mkv/.mov/.webm`` (default ``videos``)
- ``VIDEO_ENCODE_VIDEO_LIST_FILE`` — optional text file, one path per line (if set and the file exists, directory scan is not used)
- ``VIDEO_ENCODE_UPLOAD_MAX_BYTES`` — max size for ``POST /api/videos/upload`` (default ``2GiB``)
- ``VIDEO_ENCODE_UPLOAD_ANALYSIS_TIMEOUT_SEC`` — ffmpeg complexity budget when analyzing on upload (default ``600``); results are kept in an in-process dict; reset/step only read that cache (no analysis on reset)
- ``VIDEO_ENCODE_SEGMENT_SEC`` — reference clip length (default ``3.0``)
- ``VIDEO_ENCODE_STEPS_PER_SEGMENT`` — K encode attempts per timeline segment before advancing (default ``10``)
- ``VIDEO_ENCODE_FFMPEG_TIMEOUT_SEC`` — subprocess timeout for extract/VMAF (default ``600``); **libx264 encode** is capped at the **segment duration** (subprocess killed if encode exceeds that wall time)

Reward (``R = λ_Q·Q̄ − λ_T·T̄ − λ_S·S̄``):

- ``VIDEO_ENCODE_REWARD_LAMBDA_Q`` (default ``1.0``)
- ``VIDEO_ENCODE_REWARD_LAMBDA_T`` (default ``0.5``)
- ``VIDEO_ENCODE_REWARD_LAMBDA_S`` (default ``0.3``)
- ``VIDEO_ENCODE_REWARD_VMAF_MAX`` (default ``100``)
- ``VIDEO_ENCODE_REWARD_TIME_MAX_SEC`` (default ``60``)
- ``VIDEO_ENCODE_REWARD_BITRATE_MAX_KBPS`` (default ``20000``)

Logging: per-step progress uses **DEBUG**; only unusual cases use **WARNING** (e.g. empty video list, invalid index). Set log level to ``DEBUG`` to see full step traces.
"""

from __future__ import annotations

import copy
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import VideoEncodeAction, VideoEncodeObservation
    from .graders import grader_easy, grader_hard, grader_medium
    from ..segment_utils import (
        encode_segment,
        extract_segment,
        ffprobe_bitrate_kbps,
        ffprobe_duration_sec,
        ffprobe_video_size,
        ssim_score,
        vmaf_score,
    )
    from ..video_analysis import (
        analyze_segment_clip,
        analyze_video,
        load_whole_video_analysis_for_observation,
        store_whole_video_analysis,
    )
    from ..video_paths import load_video_paths
except ImportError:
    from models import VideoEncodeAction, VideoEncodeObservation
    from server.graders import grader_easy, grader_hard, grader_medium
    from segment_utils import (
        encode_segment,
        extract_segment,
        ffprobe_bitrate_kbps,
        ffprobe_duration_sec,
        ffprobe_video_size,
        ssim_score,
        vmaf_score,
    )
    from video_analysis import (
        analyze_segment_clip,
        analyze_video,
        load_whole_video_analysis_for_observation,
        store_whole_video_analysis,
    )
    from video_paths import load_video_paths

logger = logging.getLogger(__name__)


def _mean_optional(values: list[float | None]) -> float | None:
    """Arithmetic mean of non-None floats; None if empty."""
    xs = [v for v in values if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _summarize_completed_segment(
    steps: list[dict[str, Any]],
    *,
    timeline_segment_index: int,
    segment_start_sec: float,
    segment_duration_sec: float,
    steps_per_segment: int,
) -> dict[str, Any]:
    """Build prev_segment_predictions dict from K per-step metric records (means)."""
    if not steps:
        raise ValueError("steps must be non-empty")
    n = len(steps)
    return {
        "timeline_segment_index": timeline_segment_index,
        "segment_start_sec": segment_start_sec,
        "segment_duration_sec": segment_duration_sec,
        "vmaf_score": _mean_optional([s.get("vmaf_score") for s in steps]),
        "ssim_score": _mean_optional([s.get("ssim_score") for s in steps]),
        "bitrate_kbps": _mean_optional([s.get("bitrate_kbps") for s in steps]),
        "encoding_time_sec": _mean_optional([s.get("encoding_time_sec") for s in steps]),
        "crf_avg": sum(s["crf"] for s in steps) / n,
        "presets_used": [str(s["preset"]) for s in steps],
        "reward_avg": sum(s["reward"] for s in steps) / n,
        "steps_per_segment": steps_per_segment,
        "steps_averaged": n,
    }


def _encode_time_budget_sec(segment_duration_sec: float, ffmpeg_timeout_sec: float | None) -> float:
    """
    Max wall-clock seconds for libx264 encode: not longer than the reference segment duration.

    Also respects the global ffmpeg timeout when it is stricter (shorter).
    """
    seg = max(float(segment_duration_sec), 0.05)
    if ffmpeg_timeout_sec is None:
        return seg
    return min(float(ffmpeg_timeout_sec), seg)


def _timeout_from_env() -> float | None:
    raw = os.environ.get("VIDEO_ENCODE_FFMPEG_TIMEOUT_SEC", "7000")
    if raw.strip() == "":
        return None
    return float(raw)


def _reward_config_from_env() -> dict[str, float]:
    return {
        "lambda_q": float(os.environ.get("VIDEO_ENCODE_REWARD_LAMBDA_Q", "1.0")),
        "lambda_t": float(os.environ.get("VIDEO_ENCODE_REWARD_LAMBDA_T", "0.5")),
        "lambda_s": float(os.environ.get("VIDEO_ENCODE_REWARD_LAMBDA_S", "0.3")),
        "lambda_c": float(os.environ.get("VIDEO_ENCODE_REWARD_LAMBDA_C", "0.1")),
        "vmaf_max": float(os.environ.get("VIDEO_ENCODE_REWARD_VMAF_MAX", "100")),
        "bitrate_max_kbps": float(os.environ.get("VIDEO_ENCODE_REWARD_BITRATE_MAX_KBPS", "20000")),
        "grader_medium_bitrate_cap_kbps": float(os.environ.get("VIDEO_ENCODE_GRADER_MEDIUM_BITRATE_CAP", "2000.0")),
    }


class VideoEncodeEnvironment(Environment):
    """Loads a video list at init; metrics and analysis are computed in ``step``."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        *,
        videos_dir: str | Path | None = None,
        video_list_file: str | Path | None = None,
        segment_duration_sec: float | None = None,
        ffmpeg_timeout_sec: float | None = None,
        steps_per_segment: int | None = None,
    ) -> None:
        default_dir = Path(os.environ.get("VIDEO_ENCODE_VIDEOS_DIR", "videos"))
        vd = Path(videos_dir) if videos_dir is not None else default_dir

        lf: Path | None = None
        if video_list_file is not None:
            lf = Path(video_list_file)
        elif os.environ.get("VIDEO_ENCODE_VIDEO_LIST_FILE"):
            lf = Path(os.environ["VIDEO_ENCODE_VIDEO_LIST_FILE"])

        self._videos_dir = vd.expanduser().resolve()
        self._list_file = lf.expanduser().resolve() if lf is not None else None

        self._video_paths = load_video_paths(
            videos_dir=self._videos_dir if self._videos_dir.is_dir() else None,
            list_file=(
                self._list_file
                if self._list_file is not None and self._list_file.is_file()
                else None
            ),
        )
        self._analyze_new_videos()
        self._segment_duration_sec = float(
            segment_duration_sec
            if segment_duration_sec is not None
            else os.environ.get("VIDEO_ENCODE_SEGMENT_SEC", "3.0")
        )
        self._ffmpeg_timeout_sec = (
            ffmpeg_timeout_sec if ffmpeg_timeout_sec is not None else _timeout_from_env()
        )
        self._reward_cfg = _reward_config_from_env()
        self._steps_per_segment = max(
            1,
            int(
                steps_per_segment
                if steps_per_segment is not None
                else os.environ.get("VIDEO_ENCODE_STEPS_PER_SEGMENT", "10")
            ),
        )

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._current_video_idx = 0

        self._segment_start_sec = 0.0
        self._segment_index = 0
        self._step_in_segment = 0
        self._prev_segment_summary: dict[str, Any] | None = None
        self._segment_step_metrics: list[dict[str, Any]] = []

    def _analyze_new_videos(self) -> None:
        """Run ``analyze_video`` on any path not yet in the in-process cache."""
        timeout = float(os.environ.get("VIDEO_ENCODE_UPLOAD_ANALYSIS_TIMEOUT_SEC", "600"))
        for path in self._video_paths:
            if load_whole_video_analysis_for_observation(path):
                continue  # already cached
            try:
                analysis = analyze_video(path, timeout=timeout)
                store_whole_video_analysis(path, analysis)
                logger.debug("Analyzed video for observation cache: %s", path.name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not analyze %s: %s", path.name, exc)

    def _reload_video_paths_from_disk(self) -> None:
        """Rescan ``VIDEO_ENCODE_VIDEOS_DIR`` / list file so runtime uploads are visible."""
        self._video_paths = load_video_paths(
            videos_dir=self._videos_dir if self._videos_dir.is_dir() else None,
            list_file=(
                self._list_file
                if self._list_file is not None and self._list_file.is_file()
                else None
            ),
        )
        self._analyze_new_videos()

    def _observation(
        self,
        *,
        echoed_message: str,
        num_videos: int,
        current_video_index: int,
        segment_index: int,
        encode_step_index: int,
        prev_segment_predictions: dict[str, Any] | None,
        whole_video_analysis: dict[str, Any],
        segment_analysis: dict[str, Any] | None = None,
        vmaf_score: float | None,
        ssim_score: float | None,
        bitrate_kbps: float | None,
        encoding_time_sec: float | None,
        reward: float,
        done: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> VideoEncodeObservation:
        pp = copy.deepcopy(prev_segment_predictions)
        sa = copy.deepcopy(segment_analysis) if segment_analysis is not None else {}
        return VideoEncodeObservation(
            echoed_message=echoed_message,
            num_videos=num_videos,
            current_video_index=current_video_index,
            reset_index=self._reset_count,
            segment_index=segment_index,
            encode_step_index=encode_step_index,
            steps_per_segment=self._steps_per_segment,
            prev_segment_predictions=pp,
            whole_video_analysis=whole_video_analysis,
            segment_analysis=sa,
            vmaf_score=vmaf_score,
            ssim_score=ssim_score,
            bitrate_kbps=bitrate_kbps,
            encoding_time_sec=encoding_time_sec,
            done=done,
            reward=reward,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> VideoEncodeObservation:
        """New episode; cycles video index; resets timeline and K-step sub-step counter."""
        self._reload_video_paths_from_disk()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._segment_start_sec = 0.0
        self._segment_index = 0
        self._step_in_segment = 0
        self._prev_segment_summary = None
        self._segment_step_metrics = []

        n = len(self._video_paths)
        if n == 0:
            return self._observation(
                echoed_message=(
                    "No videos loaded. Add .mp4/.mkv/.mov/.webm under VIDEO_ENCODE_VIDEOS_DIR "
                    "(default: ./videos) or set VIDEO_ENCODE_VIDEO_LIST_FILE."
                ),
                num_videos=0,
                current_video_index=0,
                segment_index=0,
                encode_step_index=0,
                prev_segment_predictions=None,
                whole_video_analysis={},
                vmaf_score=None,
                ssim_score=None,
                bitrate_kbps=None,
                encoding_time_sec=None,
                reward=0.0,
            )

        self._current_video_idx = (self._reset_count - 1) % n
        ref_path = self._video_paths[self._current_video_idx]
        key = str(ref_path.resolve())
        wv = copy.deepcopy(load_whole_video_analysis_for_observation(ref_path))
        return self._observation(
            echoed_message=f"Episode ready ({self._current_video_idx + 1}/{n}): {ref_path.name}",
            num_videos=n,
            current_video_index=self._current_video_idx,
            segment_index=0,
            encode_step_index=0,
            prev_segment_predictions=None,
            whole_video_analysis=wv,
            vmaf_score=None,
            ssim_score=None,
            bitrate_kbps=None,
            encoding_time_sec=None,
            reward=0.0,
            metadata={"video_path": key, "steps_per_segment": self._steps_per_segment},
        )

    def step(self, action: VideoEncodeAction) -> VideoEncodeObservation:  # type: ignore[override]
        """K encode attempts per timeline segment; advance file cursor only after K steps."""
        self._state.step_count += 1
        n = len(self._video_paths)
        logger.debug(
            "step begin: step_count=%s reset_index=%s crf=%s preset=%s video_index=%s num_videos=%s "
            "segment_index=%s encode_step_index=%s/%s",
            self._state.step_count,
            self._reset_count,
            action.crf,
            action.preset,
            action.video_index,
            n,
            self._segment_index,
            self._step_in_segment,
            self._steps_per_segment,
        )
        if n == 0:
            logger.warning("step: no videos loaded")
            return self._observation(
                echoed_message="No videos loaded.",
                num_videos=0,
                current_video_index=0,
                segment_index=self._segment_index,
                encode_step_index=self._step_in_segment,
                prev_segment_predictions=copy.deepcopy(self._prev_segment_summary),
                whole_video_analysis={},
                vmaf_score=None,
                ssim_score=None,
                bitrate_kbps=None,
                encoding_time_sec=None,
                reward=0.0,
            )

        idx = action.video_index if action.video_index is not None else self._current_video_idx
        if idx < 0 or idx >= n:
            logger.warning("step: invalid video_index=%s (valid 0..%s)", idx, n - 1)
            return self._observation(
                echoed_message=f"Invalid video_index {idx} (need 0..{n - 1})",
                num_videos=n,
                current_video_index=self._current_video_idx,
                segment_index=self._segment_index,
                encode_step_index=self._step_in_segment,
                prev_segment_predictions=copy.deepcopy(self._prev_segment_summary),
                whole_video_analysis=copy.deepcopy(
                    load_whole_video_analysis_for_observation(
                        self._video_paths[self._current_video_idx]
                    )
                ),
                vmaf_score=None,
                ssim_score=None,
                bitrate_kbps=None,
                encoding_time_sec=None,
                reward=0.0,
            )

        ref_video = self._video_paths[idx]
        key = str(ref_video.resolve())
        whole_video_analysis = copy.deepcopy(
            load_whole_video_analysis_for_observation(ref_video)
        )

        try:
            dur = ffprobe_duration_sec(ref_video)
        except RuntimeError as e:
            logger.warning("step: ffprobe duration failed for %s: %s", key, e)
            return self._observation(
                echoed_message=str(e),
                num_videos=n,
                current_video_index=self._current_video_idx,
                segment_index=self._segment_index,
                encode_step_index=self._step_in_segment,
                prev_segment_predictions=copy.deepcopy(self._prev_segment_summary),
                whole_video_analysis=whole_video_analysis,
                vmaf_score=None,
                ssim_score=None,
                bitrate_kbps=None,
                encoding_time_sec=None,
                reward=0.0,
                metadata={"error": str(e)},
            )

        start_sec = self._segment_start_sec
        seg_len = min(self._segment_duration_sec, max(0.0, dur - start_sec))
        if seg_len < 0.1:
            logger.debug(
                "step: end of video (start_sec=%.3f dur=%.3f seg_len=%.3f)",
                start_sec,
                dur,
                seg_len,
            )
            return self._observation(
                echoed_message="No more segment data (end of video).",
                num_videos=n,
                current_video_index=self._current_video_idx,
                segment_index=self._segment_index,
                encode_step_index=self._step_in_segment,
                prev_segment_predictions=copy.deepcopy(self._prev_segment_summary),
                whole_video_analysis=whole_video_analysis,
                vmaf_score=None,
                ssim_score=None,
                bitrate_kbps=None,
                encoding_time_sec=None,
                reward=0.0,
            )

        prev_for_obs = copy.deepcopy(self._prev_segment_summary)
        timeline_idx = self._segment_index
        encode_step_this = self._step_in_segment

        to = self._ffmpeg_timeout_sec
        logger.debug(
            "step: encode segment video=%s timeline_seg=%s substep=%s start_sec=%.3f duration_sec=%.3f",
            ref_video.name,
            timeline_idx,
            encode_step_this,
            start_sec,
            seg_len,
        )
        # Populated after ref_clip extraction; {} on any pre-extraction failure path.
        seg_analysis: dict[str, Any] = {}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ref_clip = tmp_path / "ref.mp4"
            enc_path = tmp_path / "enc.mp4"
            try:
                extract_segment(ref_video, start_sec, seg_len, ref_clip, timeout=to)
                ew, eh = ffprobe_video_size(ref_clip)
                seg_analysis = analyze_segment_clip(ref_clip, timeout=to)
                encode_budget = _encode_time_budget_sec(seg_len, to)
                try:
                    enc_time = encode_segment(
                        ref_clip,
                        enc_path,
                        width=ew,
                        height=eh,
                        crf=action.crf,
                        preset=action.preset,
                        timeout=encode_budget,
                    )
                except subprocess.TimeoutExpired:
                    logger.debug(
                        "step: encode subprocess timeout at %.3fs (segment %.3fs) — reward 0, skip VMAF",
                        encode_budget,
                        seg_len,
                    )
                    return self._observation(
                        echoed_message=(
                            f"Encode exceeded segment-duration budget ({seg_len:.3f}s); "
                            "aborted, reward 0."
                        ),
                        num_videos=n,
                        current_video_index=self._current_video_idx,
                        segment_index=timeline_idx,
                        encode_step_index=encode_step_this,
                        prev_segment_predictions=prev_for_obs,
                        whole_video_analysis=whole_video_analysis,
                        segment_analysis=seg_analysis,
                        vmaf_score=None,
                        ssim_score=None,
                        bitrate_kbps=None,
                        encoding_time_sec=float(encode_budget),
                        reward=0.0,
                        metadata={
                            "encode_aborted": True,
                            "reason": "encode_timeout_vs_segment",
                            "segment_duration_sec": seg_len,
                            "encode_time_budget_sec": encode_budget,
                        },
                    )
                if enc_time > seg_len + 1e-3:
                    logger.debug(
                        "step: encode wall time %.3fs > segment %.3fs — reward 0, skip VMAF",
                        enc_time,
                        seg_len,
                    )
                    return self._observation(
                        echoed_message=(
                            f"Encode wall time ({enc_time:.3f}s) exceeded segment duration ({seg_len:.3f}s); "
                            "reward 0."
                        ),
                        num_videos=n,
                        current_video_index=self._current_video_idx,
                        segment_index=timeline_idx,
                        encode_step_index=encode_step_this,
                        prev_segment_predictions=prev_for_obs,
                        whole_video_analysis=whole_video_analysis,
                        segment_analysis=seg_analysis,
                        vmaf_score=None,
                        ssim_score=None,
                        bitrate_kbps=None,
                        encoding_time_sec=enc_time,
                        reward=0.0,
                        metadata={
                            "encode_aborted": True,
                            "reason": "encode_slower_than_segment",
                            "segment_duration_sec": seg_len,
                            "encoding_time_sec": enc_time,
                        },
                    )
                bitrate = ffprobe_bitrate_kbps(enc_path)
                vmaf = vmaf_score(enc_path, ref_clip, timeout=to)
                ssim: float | None = None
                if vmaf is None:
                    ssim = ssim_score(enc_path, ref_clip, timeout=to)
                logger.debug(
                    "step: metrics vmaf=%s ssim=%s bitrate_kbps=%s encode_time_sec=%s",
                    vmaf,
                    ssim,
                    bitrate,
                    enc_time,
                )
            except Exception as e:  # pragma: no cover - ffmpeg
                logger.exception("step encode failed")
                return self._observation(
                    echoed_message=f"Encode/measure failed: {e}",
                    num_videos=n,
                    current_video_index=self._current_video_idx,
                    segment_index=timeline_idx,
                    encode_step_index=encode_step_this,
                    prev_segment_predictions=prev_for_obs,
                    whole_video_analysis=whole_video_analysis,
                    segment_analysis=seg_analysis,
                    vmaf_score=None,
                    ssim_score=None,
                    bitrate_kbps=None,
                    encoding_time_sec=None,
                    reward=0.0,
                    metadata={"error": str(e)},
                )

        prev_crf_avg = (
            self._prev_segment_summary.get("crf_avg")
            if self._prev_segment_summary is not None
            else None
        )
        r_components: dict[str, Any] = {"task_id": action.task_id}
        if action.task_id == "easy":
            reward = grader_easy(
                vmaf=vmaf,
                encoding_time_sec=enc_time,
                segment_duration_sec=seg_len,
                encode_aborted=False,
            )
        elif action.task_id == "medium":
            reward = grader_medium(
                vmaf=vmaf,
                bitrate_kbps=bitrate,
                bitrate_cap_kbps=self._reward_cfg["grader_medium_bitrate_cap_kbps"],
                vmaf_max=self._reward_cfg["vmaf_max"],
            )
        else:  # "hard"
            # time_max_sec: use explicit env override if set, otherwise 3× the segment duration
            # so the T̄ term has meaningful gradient for typical short segments.
            _raw_tmax = os.environ.get("VIDEO_ENCODE_REWARD_TIME_MAX_SEC")
            time_max_sec = float(_raw_tmax) if _raw_tmax else 3.0 * seg_len
            reward, r_components = grader_hard(
                vmaf=vmaf,
                ssim=ssim,
                encoding_time_sec=enc_time,
                bitrate_kbps=bitrate,
                lambda_q=self._reward_cfg["lambda_q"],
                lambda_t=self._reward_cfg["lambda_t"],
                lambda_s=self._reward_cfg["lambda_s"],
                lambda_c=self._reward_cfg["lambda_c"],
                crf=action.crf,
                prev_crf_avg=prev_crf_avg,
                vmaf_max=self._reward_cfg["vmaf_max"],
                time_max_sec=time_max_sec,
                bitrate_max_kbps=self._reward_cfg["bitrate_max_kbps"],
            )
            r_components["task_id"] = action.task_id
        
        if reward <= 0.0:
            reward = 0.0001
        if reward == 1.0:
            reward = 0.9999
        logger.debug("step: reward=%s components=%s", reward, r_components)

        self._segment_step_metrics.append(
            {
                "vmaf_score": vmaf,
                "ssim_score": ssim,
                "bitrate_kbps": bitrate,
                "encoding_time_sec": enc_time,
                "crf": action.crf,
                "preset": action.preset,
                "reward": reward,
            }
        )
        self._step_in_segment += 1
        if self._step_in_segment >= self._steps_per_segment:
            logger.debug(
                "step: completed K=%s encodes for timeline segment %s; advancing timeline",
                self._steps_per_segment,
                timeline_idx,
            )
            self._prev_segment_summary = _summarize_completed_segment(
                self._segment_step_metrics,
                timeline_segment_index=timeline_idx,
                segment_start_sec=start_sec,
                segment_duration_sec=seg_len,
                steps_per_segment=self._steps_per_segment,
            )
            self._segment_step_metrics = []
            self._segment_start_sec += seg_len
            self._segment_index += 1
            self._step_in_segment = 0

        logger.debug(
            "step end: step_count=%s segment_index=%s next_encode_step_index=%s reward=%s",
            self._state.step_count,
            self._segment_index,
            self._step_in_segment,
            reward,
        )
        return self._observation(
            echoed_message="ok",
            num_videos=n,
            current_video_index=self._current_video_idx,
            segment_index=timeline_idx,
            encode_step_index=encode_step_this,
            prev_segment_predictions=prev_for_obs,
            whole_video_analysis=whole_video_analysis,
            segment_analysis=seg_analysis,
            vmaf_score=vmaf,
            ssim_score=ssim,
            bitrate_kbps=bitrate,
            encoding_time_sec=enc_time,
            reward=reward,
            metadata={
                "video_path": key,
                "segment_sec": seg_len,
                "segment_start_sec": start_sec,
                "encode_width": ew,
                "encode_height": eh,
                "crf": action.crf,
                "preset": action.preset,
                "step": self._state.step_count,
                "steps_per_segment": self._steps_per_segment,
                "reward_components": r_components,
            },
        )

    @property
    def state(self) -> State:
        return self._state
