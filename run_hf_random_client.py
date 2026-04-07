#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Connect to a deployed Hugging Face Space (or local server) over WebSocket and run
``reset`` + ``step`` with random encoding actions.

By default the client uses the **hosted Hugging Face Space** (no flags required).
Override with ``--base-url``, ``VIDEO_ENCODE_BASE_URL``, ``--local``, or ``--repo-id``.

Examples::

    # Default: hosted Space (same as --base-url below)
    uv run hf-random-client --num-steps 5

    # Explicit hosted URL
    uv run python run_hf_random_client.py \\
      --base-url https://keshav142-video_encode.hf.space --num-steps 5

    # Local server
    uv run python run_hf_random_client.py --local --num-steps 3

    # Pull Space image via OpenEnv (requires Docker) instead of hosted HTTP
    uv run python run_hf_random_client.py --repo-id keshav142/video_encode --num-steps 3

``--repo-id`` **starts a new container** from the HF registry and waits for ``/health`` (it does
not attach to a container you started yourself). For a server you already run locally, use
``--docker-container <id>`` (looks up the host port with ``docker port``) or
``--base-url http://127.0.0.1:<mapped-port>``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from typing import Optional

from openenv.core.containers.runtime.providers import LocalDockerProvider

from video_encode import VideoEncodeAction, VideoEncodeEnv

# Default when no --base-url / --local / VIDEO_ENCODE_BASE_URL (hosted env on Hugging Face).
_DEFAULT_HF_SPACE_BASE_URL = "https://keshav142-video_encode.hf.space"

class ExtendedReadyLocalDockerProvider(LocalDockerProvider):
    """
    Same as OpenEnv's LocalDockerProvider, but ``from_env`` only passes ``base_url`` to
    ``wait_for_ready``; the default 30s is too short for slow-starting env images (uv + ffmpeg).
    """

    def __init__(self, ready_timeout_s: float = 600.0) -> None:
        super().__init__()
        self._ready_timeout_s = float(ready_timeout_s)

    def wait_for_ready(self, base_url: str, timeout_s: float | None = None) -> None:
        t = self._ready_timeout_s if timeout_s is None else float(timeout_s)
        return super().wait_for_ready(base_url, timeout_s=t)


def _base_url_from_docker_container(container: str) -> str:
    """
    Resolve ``http://127.0.0.1:<host-port>`` for a running container's container port 8000.

    Uses ``docker port <container> 8000/tcp`` (same mapping OpenEnv prints as localhost:63459).
    """
    r = subprocess.run(
        ["docker", "port", container, "8000/tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
        raise RuntimeError(f"docker port {container} 8000/tcp failed: {msg}")
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(
            f"No host port for {container} 8000/tcp — is the container running with -p ...:8000?"
        )
    line = lines[0]
    # "0.0.0.0:63459" or "[::]:63459"
    if "]:" in line:
        host_port = int(line.split("]:")[1])
    else:
        host_port = int(line.rsplit(":", 1)[1])
    return f"http://127.0.0.1:{host_port}"


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


def _random_action(rng: random.Random, num_videos: int) -> VideoEncodeAction:
    crf = rng.randint(18, 28)
    preset = rng.choice(_X264_PRESETS)
    video_index: int | None = None
    if num_videos > 0:
        video_index = rng.randrange(num_videos)
    return VideoEncodeAction(crf=crf, preset=preset, video_index=video_index)


async def _run(
    *,
    base_url: Optional[str],
    repo_id: Optional[str],
    use_docker: bool,
    from_env_ready_timeout_s: float,
    num_steps: int,
    seed: Optional[int],
    quiet: bool,
) -> None:
    rng = random.Random(seed)

    if repo_id:
        if use_docker:
            if not quiet:
                print(
                    "Starting a new container from registry.hf.space (not connecting to other "
                    f"running containers). /health wait up to {from_env_ready_timeout_s:.0f}s.",
                    flush=True,
                )
            provider = ExtendedReadyLocalDockerProvider(ready_timeout_s=from_env_ready_timeout_s)
            env = await VideoEncodeEnv.from_env(repo_id, use_docker=True, provider=provider)
        else:
            env = await VideoEncodeEnv.from_env(repo_id, use_docker=False)
    else:
        assert base_url is not None
        if not quiet:
            print(f"Using server: {base_url}", flush=True)
        env = VideoEncodeEnv(base_url=base_url)
        await env.connect()

    last_reward: float | None = None
    steps_done = 0
    try:
        reset_result = await env.reset()
        obs = reset_result.observation
        if not quiet:
            print("--- reset ---")
            print(f"echoed_message: {obs.echoed_message}")
            print(f"num_videos: {obs.num_videos}")
            print(f"reset_index: {obs.reset_index}")
            print(json.dumps(obs.model_dump(), indent=2, default=str))

        for i in range(num_steps):
            action = _random_action(rng, obs.num_videos)
            if not quiet:
                print(
                    f"\n--- step {i + 1}/{num_steps} action: crf={action.crf} preset={action.preset} video_index={action.video_index} ---"
                )
            step_result = await env.step(action)
            obs = step_result.observation
            last_reward = step_result.reward
            steps_done = i + 1
            if not quiet:
                print(
                    f"reward: {step_result.reward} encode_step_index: {obs.encode_step_index} segment_index: {obs.segment_index}"
                )
                em = obs.echoed_message
                print(f"echoed_message: {em[:200]}..." if len(em) > 200 else f"echoed_message: {em}")
            if step_result.done:
                if not quiet:
                    print("done=True, stopping early")
                break
        if quiet:
            print(
                f"ok: steps={steps_done} last_reward={last_reward}",
                flush=True,
            )
    finally:
        await env.close()


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Random-action client for Video Encode env. "
            "Defaults to the hosted Hugging Face Space; use --local or --base-url to override."
        ),
    )
    p.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help=(
            "Server root URL (https://…hf.space or http://localhost:8000). "
            "If omitted: use VIDEO_ENCODE_BASE_URL, else the default hosted Space URL."
        ),
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Connect to http://127.0.0.1:8000 (overrides default hosted URL unless --base-url is set).",
    )
    p.add_argument(
        "--docker-container",
        default=None,
        metavar="ID_OR_NAME",
        help=(
            "Connect to an already-running container: run `docker port ID 8000/tcp` and use "
            "that host port (e.g. b553c2979c1e → http://127.0.0.1:63459). Does not start a new container."
        ),
    )
    p.add_argument(
        "--repo-id",
        default=None,
        metavar="USER/NAME",
        help=(
            "Hugging Face Space id (e.g. keshav142/video_encode). "
            "Runs the env via OpenEnv from_env() (Docker by default), not the hosted HTTP URL."
        ),
    )
    p.add_argument(
        "--no-docker",
        action="store_true",
        help="With --repo-id, use UVProvider instead of pulling the Space Docker image.",
    )
    p.add_argument(
        "--ready-timeout",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "With --repo-id and Docker: seconds to wait for GET /health (default 600, or "
            "VIDEO_ENCODE_FROM_ENV_READY_TIMEOUT). Increase if the image is slow to start."
        ),
    )
    p.add_argument("--num-steps", type=int, default=5, help="Number of random steps after reset")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible actions")
    p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Minimal output (one summary line; server logs use DEBUG for per-step detail).",
    )
    args = p.parse_args()

    repo_id = (args.repo_id or os.environ.get("VIDEO_ENCODE_REPO_ID") or "").strip() or None
    docker_container = (args.docker_container or "").strip() or None

    if repo_id and (args.base_url or args.local or docker_container):
        print(
            "Error: use --repo-id alone, or use --base-url / --local / --docker-container "
            "(not --repo-id).",
            file=sys.stderr,
        )
        return 2

    if docker_container and (args.base_url or args.local):
        print(
            "Error: use either --docker-container or --base-url / --local, not both.",
            file=sys.stderr,
        )
        return 2

    if repo_id:
        base_url: str | None = None
    elif docker_container:
        try:
            base_url = _base_url_from_docker_container(docker_container)
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    elif args.base_url:
        base_url = args.base_url
    elif args.local:
        base_url = "http://127.0.0.1:8000"
    else:
        base_url = os.environ.get("VIDEO_ENCODE_BASE_URL", "").strip() or _DEFAULT_HF_SPACE_BASE_URL

    ready_raw = os.environ.get("VIDEO_ENCODE_FROM_ENV_READY_TIMEOUT", "600")
    try:
        default_ready = float(ready_raw)
    except ValueError:
        default_ready = 600.0
    from_env_ready = float(args.ready_timeout) if args.ready_timeout is not None else default_ready

    asyncio.run(
        _run(
            base_url=base_url,
            repo_id=repo_id,
            use_docker=not args.no_docker,
            from_env_ready_timeout_s=from_env_ready,
            num_steps=max(0, args.num_steps),
            seed=args.seed,
            quiet=args.quiet,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
