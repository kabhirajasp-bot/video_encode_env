# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Video Encode Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import VideoEncodeAction, VideoEncodeObservation


class VideoEncodeEnv(
    EnvClient[VideoEncodeAction, VideoEncodeObservation, State]
):
    """
    Client for the Video Encode Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with VideoEncodeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(VideoEncodeAction(crf=23))
    """

    def _step_payload(self, action: VideoEncodeAction) -> Dict[str, Any]:
        """Convert VideoEncodeAction to JSON payload for step message."""
        payload: Dict[str, Any] = {
            "crf": action.crf,
            "preset": action.preset,
        }
        if action.video_index is not None:
            payload["video_index"] = action.video_index
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[VideoEncodeObservation]:
        """Parse server response into StepResult[VideoEncodeObservation]."""
        obs_data = payload.get("observation", {})
        observation = VideoEncodeObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            num_videos=obs_data.get("num_videos", 0),
            current_video_index=obs_data.get("current_video_index", 0),
            reset_index=obs_data.get("reset_index", 0),
            segment_index=obs_data.get("segment_index", 0),
            encode_step_index=obs_data.get("encode_step_index", 0),
            steps_per_segment=obs_data.get("steps_per_segment", 10),
            prev_segment_predictions=obs_data.get("prev_segment_predictions"),
            whole_video_analysis=obs_data.get("whole_video_analysis") or {},
            vmaf_score=obs_data.get("vmaf_score"),
            ssim_score=obs_data.get("ssim_score"),
            bitrate_kbps=obs_data.get("bitrate_kbps"),
            encoding_time_sec=obs_data.get("encoding_time_sec"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
