# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Discover video file paths from a directory or a line-based list file."""

from __future__ import annotations

from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".webm")


def list_video_files(videos_dir: Path) -> list[Path]:
    """Sorted list of video files directly under ``videos_dir`` (non-recursive)."""
    if not videos_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(p)
    return out


def load_video_paths_from_file(list_file: Path) -> list[Path]:
    """One absolute or relative path per line; ``#`` starts a comment; skips missing files."""
    if not list_file.is_file():
        return []
    base = list_file.parent
    out: list[Path] = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (base / p).resolve()
        if p.is_file():
            out.append(p)
    return out


def load_video_paths(
    *,
    videos_dir: Path | None = None,
    list_file: Path | None = None,
) -> list[Path]:
    """
    Resolve a list of video paths: prefer ``list_file`` if given and present,
    else scan ``videos_dir``.
    """
    if list_file is not None and list_file.is_file():
        return load_video_paths_from_file(list_file)
    if videos_dir is not None and videos_dir.is_dir():
        return list_video_files(videos_dir)
    return []
