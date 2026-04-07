# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Video Encode Environment.

This module creates an HTTP server that exposes the VideoEncodeEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /api/videos: List videos under ``VIDEO_ENCODE_VIDEOS_DIR``
    - POST /api/videos/upload: Multipart upload into that directory (then call ``/reset``)
    - DELETE /api/videos/{filename}: Remove an uploaded video file from that directory
    - POST /api/videos/delete: Same removal with JSON body ``{"filename": "clip.mp4"}`` (use if DELETE returns 404)

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import VideoEncodeAction, VideoEncodeObservation
    from .video_encode_environment import VideoEncodeEnvironment
    from .video_upload import build_video_upload_router
except ModuleNotFoundError:
    from models import VideoEncodeAction, VideoEncodeObservation
    from server.video_encode_environment import VideoEncodeEnvironment
    from server.video_upload import build_video_upload_router


# Create the app with web interface and README integration
app = create_app(
    VideoEncodeEnvironment,
    VideoEncodeAction,
    VideoEncodeObservation,
    env_name="video_encode",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)
app.include_router(build_video_upload_router())


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run server
        uv run server -- --port 8001

    ``HOST`` and ``PORT`` environment variables are used (Docker / Hugging Face
    Spaces set ``PORT``). ``--port`` overrides ``PORT`` when passed.

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn video_encode.server.app:app --workers 4
    """
    import argparse
    import os

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    args, _ = parser.parse_known_args()
    host = os.environ.get("HOST", "0.0.0.0")
    if args.port is not None:
        port = args.port
    else:
        port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
