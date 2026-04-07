#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: reset + episode loop over sample_video; prints observation after each step."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel

# libx264 -preset values (same family as VideoEncodeAction; omit placebo for smoke runtime)
_X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)


def _print_observation(obs: BaseModel, title: str) -> None:
    """Pretty-print main observation fields (JSON for nested dicts)."""
    d = obs.model_dump()
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for key in (
        "echoed_message",
        "num_videos",
        "current_video_index",
        "reset_index",
        "segment_index",
        "encode_step_index",
        "steps_per_segment",
        "reward",
        "done",
    ):
        if key in d:
            print(f"  {key}: {d[key]}")
    for key in ("prev_segment_predictions",):
        if key in d and d[key] is not None:
            print(f"  {key}:\n{json.dumps(d[key], indent=4)}")
        elif key in d:
            print(f"  {key}: {d[key]}")
    for key in ("whole_video_analysis",):
        if d.get(key):
            print(f"  {key}:\n{json.dumps(d[key], indent=2)}")
    for key in ("vmaf_score", "ssim_score", "bitrate_kbps", "encoding_time_sec"):
        if key in d:
            print(f"  {key}: {d[key]}")
    md = d.get("metadata") or {}
    if md:
        rc = md.pop("reward_components", None)
        if rc is not None:
            print(f"  reward_components:\n{json.dumps(rc, indent=4)}")
        if md:
            print(f"  metadata:\n{json.dumps(md, indent=2)}")


def main() -> int:
    root = Path(__file__).resolve().parent
    default_video = root.parent / "sample_video.mp4"

    p = argparse.ArgumentParser(
        description=(
            "Run VideoEncodeEnvironment: reset then step until the video is exhausted. "
            "Each timeline segment runs K encodes (steps_per_segment) before advancing; "
            "no fixed max step count—stops at end of video."
        ),
    )
    p.add_argument(
        "--video",
        type=Path,
        default=default_video,
        help=f"Path to input .mp4 (default: {default_video})",
    )
    p.add_argument("--segment-sec", type=float, default=10.0, help="Reference clip length")
    p.add_argument(
        "--steps-per-segment",
        type=int,
        default=10,
        help="K encode attempts per timeline segment before advancing",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducible random crf/preset each step (default: nondeterministic)",
    )
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg and ffprobe must be on PATH (e.g. brew install ffmpeg)", file=sys.stderr)
        return 1

    video = args.video.resolve()
    if not video.is_file():
        print(f"Video not found: {video}", file=sys.stderr)
        print("Hint: place sample_video.mp4 under rl-hackathon/ or pass --video", file=sys.stderr)
        return 1

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"{video}\n")
        list_path = Path(f.name)

    try:
        from video_encode.server.video_encode_environment import VideoEncodeEnvironment
        from video_encode.models import VideoEncodeAction

        def sample_action() -> VideoEncodeAction:
            return VideoEncodeAction(
                crf=random.randint(0, 51),
                preset=random.choice(_X264_PRESETS),
            )

        env = VideoEncodeEnvironment(
            video_list_file=list_path,
            segment_duration_sec=args.segment_sec,
            steps_per_segment=args.steps_per_segment,
        )
        r = env.reset()
        _print_observation(r, "OBSERVATION after reset()")

        last_ok = True
        step_i = 0
        while True:
            step_i += 1
            action = sample_action()
            print(f"\n  action: crf={action.crf}, preset={action.preset!r}")
            o = env.step(action)
            _print_observation(o, f"OBSERVATION after step({step_i})")
            if o.echoed_message != "ok":
                last_ok = False
            if "No more segment data" in (o.echoed_message or ""):
                print("\n(Stopping: end of video — all segments processed.)")
                break
            if o.echoed_message and "Encode/measure failed" in o.echoed_message:
                break
        return 0 if last_ok else 1
    finally:
        list_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
