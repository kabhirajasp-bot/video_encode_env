# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset sampling, segmentation, and ffmpeg grid collection.

Stores (segment_features, params) -> (bitrate_kbps, vmaf, ssim, encoding_time_sec)
as JSON lines in ``grid_records.jsonl`` under ``data_root``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .segment_utils import (
    build_segment_features,
    encode_segment,
    extract_segment,
    ffprobe_bitrate_kbps,
    ffprobe_duration_sec,
    ssim_score,
    vmaf_score,
)
from .video_paths import VIDEO_EXTENSIONS, list_video_files

logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = Path(os.environ.get("VIDEO_ENCODE_DATA_ROOT", "data"))
DEFAULT_VIDEOS_DIR = Path(os.environ.get("VIDEO_ENCODE_VIDEOS_DIR", "videos"))

GRID_CRF = (18, 22, 26, 30)
GRID_PRESETS = ("fast", "medium", "slow")
# width x height for 360p, 720p, 1080p
GRID_RESOLUTIONS: tuple[tuple[int, int, str], ...] = (
    (640, 360, "360p"),
    (1280, 720, "720p"),
    (1920, 1080, "1080p"),
)

MANIFEST_NAME = "segments_manifest.json"
RECORDS_NAME = "grid_records.jsonl"


def sample_videos(paths: list[Path], max_videos: int, seed: int) -> list[Path]:
    if not paths:
        return []
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    return shuffled[: min(max_videos, len(shuffled))]


def iter_segment_windows(
    video_duration_sec: float,
    *,
    min_seg: float,
    max_seg: float,
    rng: random.Random,
) -> Iterator[tuple[float, float]]:
    """Yield (start_sec, duration_sec) until duration exhausted."""
    t = 0.0
    while t + min_seg <= video_duration_sec + 1e-6:
        dur = rng.uniform(min_seg, max_seg)
        if t + dur > video_duration_sec:
            dur = max(min_seg, video_duration_sec - t)
        if dur < min_seg - 1e-6:
            break
        yield t, dur
        t += dur


def segment_source_videos(
    video_paths: list[Path],
    segments_dir: Path,
    *,
    min_segment_sec: float,
    max_segment_sec: float,
    max_segments_per_video: int | None,
    max_total_segments: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Cut each source into 2–4s (default) segments; return manifest entries.
    """
    rng = random.Random(seed)
    manifest: list[dict[str, Any]] = []
    total = 0

    for src in video_paths:
        vid = src.stem
        dur = ffprobe_duration_sec(src)
        seg_idx = 0
        for start, length in iter_segment_windows(
            dur,
            min_seg=min_segment_sec,
            max_seg=max_segment_sec,
            rng=rng,
        ):
            if max_total_segments is not None and total >= max_total_segments:
                return manifest
            if max_segments_per_video is not None and seg_idx >= max_segments_per_video:
                break
            out = segments_dir / vid / f"seg_{seg_idx:05d}.mp4"
            try:
                extract_segment(src, start, length, out)
            except RuntimeError as e:
                logger.warning("skip segment %s @ %.2fs: %s", src, start, e)
                seg_idx += 1
                continue
            rel = str(out.relative_to(segments_dir.parent))
            manifest.append(
                {
                    "video_id": vid,
                    "segment_index": seg_idx,
                    "path": str(out.resolve()),
                    "relative_path": rel,
                    "source_video": str(src.resolve()),
                    "start_sec": start,
                    "duration_sec": length,
                }
            )
            seg_idx += 1
            total += 1
            if max_total_segments is not None and total >= max_total_segments:
                return manifest
        if max_total_segments is not None and total >= max_total_segments:
            break

    return manifest


def grid_param_iter() -> Iterator[dict[str, Any]]:
    for crf in GRID_CRF:
        for preset in GRID_PRESETS:
            for w, h, label in GRID_RESOLUTIONS:
                yield {
                    "crf": crf,
                    "preset": preset,
                    "width": w,
                    "height": h,
                    "resolution_label": label,
                }


def run_grid_on_segment(
    segment_path: Path,
    segment_features: dict[str, Any],
    *,
    ffmpeg_timeout_sec: float | None = 600.0,
) -> list[dict[str, Any]]:
    """Encode segment for each grid cell; measure bitrate, VMAF, SSIM, encode time."""
    records: list[dict[str, Any]] = []
    for params in grid_param_iter():
        with tempfile.TemporaryDirectory() as tmp:
            enc_path = Path(tmp) / "encoded.mp4"
            try:
                enc_time = encode_segment(
                    segment_path,
                    enc_path,
                    width=params["width"],
                    height=params["height"],
                    crf=params["crf"],
                    preset=params["preset"],
                    timeout=ffmpeg_timeout_sec,
                )
            except RuntimeError as e:
                logger.warning("encode failed %s %s: %s", segment_path, params, e)
                continue
            try:
                bitrate = ffprobe_bitrate_kbps(enc_path)
            except RuntimeError:
                bitrate = 0.0
            vmaf = vmaf_score(enc_path, segment_path, timeout=ffmpeg_timeout_sec)
            ssim: float | None = None
            if vmaf is None:
                ssim = ssim_score(enc_path, segment_path, timeout=ffmpeg_timeout_sec)
            row = {
                "segment_features": segment_features,
                "params": {
                    "crf": params["crf"],
                    "preset": params["preset"],
                    "width": params["width"],
                    "height": params["height"],
                    "resolution_label": params["resolution_label"],
                },
                "bitrate_kbps": bitrate,
                "vmaf": vmaf,
                "ssim": ssim,
                "encoding_time_sec": enc_time,
            }
            records.append(row)
    return records


def collect_dataset(
    *,
    data_root: Path,
    videos_dir: Path,
    max_videos: int = 50,
    min_segment_sec: float = 2.0,
    max_segment_sec: float = 4.0,
    max_segments_per_video: int | None = 10,
    max_total_segments: int | None = 150,
    seed: int = 42,
    append_records: bool = False,
    skip_if_manifest_exists: bool = False,
) -> dict[str, Any]:
    """
    Sample videos, segment, run CRF × preset × resolution grid, append JSONL.

    Returns summary dict with paths and counts.
    """
    data_root = data_root.resolve()
    videos_dir = videos_dir.resolve()
    segments_dir = data_root / "segments"
    records_path = data_root / RECORDS_NAME
    manifest_path = data_root / MANIFEST_NAME

    data_root.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    if skip_if_manifest_exists and manifest_path.is_file():
        logger.info("Manifest exists, skipping segmentation/collection: %s", manifest_path)
        return load_dataset_stats(data_root)

    all_videos = list_video_files(videos_dir)
    if not all_videos:
        raise FileNotFoundError(
            f"No video files ({', '.join(VIDEO_EXTENSIONS)}) in {videos_dir}. "
            "Add sources or set VIDEO_ENCODE_VIDEOS_DIR."
        )

    sampled = sample_videos(all_videos, max_videos, seed)
    manifest = segment_source_videos(
        sampled,
        segments_dir,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        max_segments_per_video=max_segments_per_video,
        max_total_segments=max_total_segments,
        seed=seed,
    )

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mode = "a" if append_records and records_path.is_file() else "w"
    n_written = 0
    with records_path.open(mode, encoding="utf-8") as f:
        for entry in manifest:
            sp = Path(entry["path"])
            if not sp.is_file():
                logger.warning("missing segment file %s", sp)
                continue
            feats = build_segment_features(
                sp,
                video_id=entry["video_id"],
                segment_index=entry["segment_index"],
            )
            for row in run_grid_on_segment(sp, feats):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1

    return {
        "data_root": str(data_root),
        "videos_dir": str(videos_dir),
        "num_videos_sampled": len(sampled),
        "num_segments": len(manifest),
        "num_grid_records": n_written,
        "manifest_path": str(manifest_path),
        "records_path": str(records_path),
    }


def load_dataset_stats(data_root: Path | str) -> dict[str, Any]:
    """Read manifest + count JSONL lines without loading full file."""
    root = Path(data_root).resolve()
    manifest_path = root / MANIFEST_NAME
    records_path = root / RECORDS_NAME
    manifest: list[Any] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_lines = 0
    if records_path.is_file():
        with records_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n_lines += 1
    video_ids = {m.get("video_id") for m in manifest if isinstance(m, dict)}
    return {
        "data_root": str(root),
        "num_videos_sampled": len(video_ids),
        "num_segments": len(manifest),
        "num_grid_records": n_lines,
        "manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "records_path": str(records_path) if records_path.is_file() else "",
        "dataset_ready": manifest_path.is_file() and records_path.is_file() and n_lines > 0,
    }


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Collect ffmpeg grid metrics for video segments.")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    p.add_argument("--max-videos", type=int, default=50)
    p.add_argument("--min-segment-sec", type=float, default=2.0)
    p.add_argument("--max-segment-sec", type=float, default=4.0)
    p.add_argument("--max-segments-per-video", type=int, default=10)
    p.add_argument("--max-total-segments", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append", action="store_true", help="Append to JSONL instead of overwrite.")
    args = p.parse_args()
    summary = collect_dataset(
        data_root=args.data_root,
        videos_dir=args.videos_dir,
        max_videos=args.max_videos,
        min_segment_sec=args.min_segment_sec,
        max_segment_sec=args.max_segment_sec,
        max_segments_per_video=args.max_segments_per_video,
        max_total_segments=args.max_total_segments,
        seed=args.seed,
        append_records=args.append,
        skip_if_manifest_exists=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
