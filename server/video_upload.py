# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP routes to upload videos into ``VIDEO_ENCODE_VIDEOS_DIR`` (e.g. Hugging Face Space)."""

from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

try:
    from ..video_analysis import analyze_video, forget_whole_video_analysis, store_whole_video_analysis
    from ..video_paths import VIDEO_EXTENSIONS, list_video_files
except ImportError:
    from video_analysis import analyze_video, forget_whole_video_analysis, store_whole_video_analysis
    from video_paths import VIDEO_EXTENSIONS, list_video_files

logger = logging.getLogger(__name__)

# ``analyze_video`` (ffmpeg luma/complexity) can be slow; outer wait allows ffprobe + I/O headroom.
_DEFAULT_UPLOAD_ANALYSIS_TIMEOUT_SEC = 600.0
_UPLOAD_WAIT_HEADROOM_SEC = 180.0


def videos_dir() -> Path:
    return Path(os.environ.get("VIDEO_ENCODE_VIDEOS_DIR", "videos")).expanduser().resolve()


def default_upload_max_bytes() -> int:
    raw = os.environ.get("VIDEO_ENCODE_UPLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024))
    try:
        return max(1, int(raw))
    except ValueError:
        return 2 * 1024 * 1024 * 1024


def _safe_filename(name: str | None) -> str:
    if not name:
        raise HTTPException(status_code=400, detail="Filename required")
    base = Path(name).name
    if not base or base in (".", "..") or "/" in base or "\\" in base:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suf = Path(base).suffix.lower()
    if suf not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension {suf!r}; allowed: {', '.join(VIDEO_EXTENSIONS)}",
        )
    return base


class DeleteVideoBody(BaseModel):
    """JSON body for ``POST /api/videos/delete`` (same semantics as ``DELETE /api/videos/{filename}``)."""

    filename: str = Field(..., description="Basename only, e.g. clip.mp4")


def _delete_uploaded_video_impl(filename: str) -> dict:
    """Remove one file under ``VIDEO_ENCODE_VIDEOS_DIR`` and drop its analysis cache."""
    root = videos_dir()
    name = _safe_filename(filename)
    root_res = root.resolve()
    dest = (root / name).resolve()
    try:
        dest.relative_to(root_res)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes videos directory") from None
    if not dest.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    forget_whole_video_analysis(dest)
    dest.unlink()
    logger.info("Deleted uploaded video %s", name)
    return {
        "ok": True,
        "deleted": name,
        "hint": "Call POST /reset on the environment (or reconnect) to refresh the video list.",
    }


def build_video_upload_router() -> APIRouter:
    router = APIRouter(tags=["videos"])

    @router.get("/api/videos")
    async def list_uploaded_videos() -> dict:
        """List video files in ``VIDEO_ENCODE_VIDEOS_DIR`` (non-recursive)."""
        root = videos_dir()
        if not root.is_dir():
            return {"videos_dir": str(root), "exists": False, "files": []}
        files = [p.name for p in list_video_files(root)]
        return {"videos_dir": str(root), "exists": True, "count": len(files), "files": files}

    @router.post("/api/videos/upload")
    async def upload_video(file: UploadFile = File(...)) -> dict:
        """
        Save an uploaded file into ``VIDEO_ENCODE_VIDEOS_DIR``.

        Call ``reset()`` on the environment afterward (or start a new session) so the
        new file is included in the episode video list.
        """
        name = _safe_filename(file.filename)
        root = videos_dir()
        root.mkdir(parents=True, exist_ok=True)

        dest = root / name
        if dest.is_file():
            forget_whole_video_analysis(dest)
        max_bytes = default_upload_max_bytes()
        written = 0

        try:
            with dest.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_bytes:
                        dest.unlink(missing_ok=True)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds VIDEO_ENCODE_UPLOAD_MAX_BYTES ({max_bytes})",
                        )
                    out.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

        raw_to = os.environ.get(
            "VIDEO_ENCODE_UPLOAD_ANALYSIS_TIMEOUT_SEC",
            str(_DEFAULT_UPLOAD_ANALYSIS_TIMEOUT_SEC),
        )
        try:
            analysis_timeout = float(raw_to)
        except ValueError:
            analysis_timeout = _DEFAULT_UPLOAD_ANALYSIS_TIMEOUT_SEC

        analysis_error: str | None = None
        try:
            analysis = await asyncio.wait_for(
                asyncio.to_thread(
                    partial(analyze_video, dest, complexity_timeout_sec=analysis_timeout),
                ),
                timeout=analysis_timeout + _UPLOAD_WAIT_HEADROOM_SEC,
            )
            store_whole_video_analysis(dest, analysis)
            logger.info("Stored in-memory video analysis for %s", name)
        except asyncio.TimeoutError:
            analysis_error = (
                f"analyze_video exceeded {analysis_timeout + _UPLOAD_WAIT_HEADROOM_SEC}s"
            )
            logger.warning("Upload analysis timeout for %s: %s", name, analysis_error)
        except Exception as e:
            analysis_error = str(e)
            logger.warning("Upload analysis failed for %s: %s", name, e, exc_info=True)

        out: dict[str, object] = {
            "ok": True,
            "path": str(dest.resolve()),
            "filename": name,
            "size_bytes": written,
            "hint": "Call POST /reset on the environment (or reconnect) to refresh the video list.",
        }
        if analysis_error is not None:
            out["analysis_error"] = analysis_error
        return out

    @router.post("/api/videos/delete")
    async def delete_uploaded_video_post(body: DeleteVideoBody) -> dict:
        """
        Same as ``DELETE /api/videos/{filename}``, but uses JSON so it works behind
        proxies that only allow GET/POST and for older images that lack the DELETE route.
        """
        return _delete_uploaded_video_impl(body.filename)

    @router.delete("/api/videos/{filename}")
    async def delete_uploaded_video(filename: str) -> dict:
        """
        Remove a video file under ``VIDEO_ENCODE_VIDEOS_DIR`` and its in-memory analysis cache.

        Filename must be a single basename (same rules as upload). Call ``reset`` afterward
        so the environment rescans the directory.
        """
        return _delete_uploaded_video_impl(filename)

    return router
