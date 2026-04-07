#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference Script — Video Encode Environment
============================================

LLM setup (Hugging Face free model ``google/flan-t5-small``)
- Uses the **OpenAI Python SDK** (``openai`` package) only as an **HTTP client**; it does **not**
  call OpenAI's ``api.openai.com`` unless you override ``API_BASE_URL``.
- Default ``API_BASE_URL`` is **Hugging Face's OpenAI-compatible router**:
  ``https://router.huggingface.co/v1`` — this is where the free HF-hosted model runs.
- **Authentication:** set a **Hugging Face user token** (read is enough for many models):
  ``export HF_TOKEN=hf_...`` from https://huggingface.co/settings/tokens
- ``OPENAI_API_KEY`` is accepted as an **alias** for the same secret (the SDK expects that name
  in many setups); for this project it must still be your **HF token** when using the HF router,
  not an OpenAI ``sk-...`` key.
- ``MODEL_NAME`` must be the **exact Hugging Face model id** (repo name), e.g.
  ``google/flan-t5-small`` or ``Qwen/Qwen1.5-7B-Chat``. Do **not** use human labels like
  ``Qwen1.5-7B/14B`` (spaces/slash between two sizes are not a valid id). The inference router
  returns ``model_not_found`` if the id is wrong. If the id is correct but you see
  ``model_not_supported`` / "not supported by any provider you have enabled", the model exists on
  the Hub but **no inference provider is enabled for your account** (or that size is not offered on
  the router); pick a model listed for Serverless / Router on `huggingface.co`, enable providers in
  HF account settings, or point ``API_BASE_URL`` at your own OpenAI-compatible server.

Other environment variables
- ``VIDEO_ENCODE_ACTION_PARSE_LAST_LINES`` — When finding delimited JSON, search the last N lines first
  (default ``7``), then the full message.
- ``VIDEO_ENCODE_HF_URL`` or ``VIDEO_ENCODE_BASE_URL`` — Video Encode **environment** server (e.g. your
  Hugging Face Space ``https://<user>-<space>.hf.space`` — use **hyphens** in the hostname, not
  underscores (underscores cause TLS errors). Used when you do **not** pass ``--local`` or ``--hf-url``.
  Override default with ``VIDEO_ENCODE_DEFAULT_HF_URL``.
- ``LOCAL_IMAGE_NAME`` / ``IMAGE_NAME`` — Optional Docker image for ``from_docker_image``.
- ``MAX_TOKENS`` — Completion cap (default ``1024``); echoed in system/user prompts so the model keeps JSON within budget.
- ``INFERENCE_JSON_OBJECT_MODE`` — Default ``0``. Keep off when using delimiter markers: ``json_object``
  mode forces a bare JSON reply and **prevents** the ``<<<VIDEO_ENCODE_ACTION_JSON>>>`` lines. Set to ``1``
  only if you use a different parser (not recommended with current client).
- ``INFERENCE_DISABLE_THINKING`` — Default ``1``. When set, the client sends ``enable_thinking: false`` in
  ``extra_body`` (Qwen3 / OpenAI-compatible) so the model prefers a direct reply over long reasoning.
  Set to ``0`` if your provider rejects unknown fields or you want thinking enabled.

STDOUT FORMAT (exactly these line types)
- ``[ENV]`` once: how the RL env is reached (Space URL, ``repo_id``, local, or Docker).
- ``[START] task=<task> env=<benchmark> model=<model_name>`` once at episode begin.
- ``[USER_PROMPT] step=<n>`` then the full user message (before each model call). Disable with
  ``VIDEO_ENCODE_PRINT_USER_PROMPT=0`` if logs are too large.
- ``[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`` after each ``env.step()``.
- ``[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>`` after ``env.close()`` (always, even on error).

If the model call fails or the reply is not valid JSON, the client uses fixed defaults
(``crf=23``, ``preset=medium``, ``video_index=0``), so those fields repeat until inference works.

Uses OpenAI client + observation JSON in the user prompt. Assistant text is taken preferentially
from ``message.content``, then from ``reasoning`` / ``reasoning_content`` / ``text`` if ``content`` is
empty (common on some HF routers). Delimited JSON must appear in that combined assistant text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import textwrap
from typing import Any, List, Optional

from openai import OpenAI
from openenv.core.containers.runtime.providers import LocalDockerProvider
from pydantic import ValidationError

from video_encode import VideoEncodeAction, VideoEncodeEnv

# --- LLM: HF router + google/flan-t5-small (OpenAI SDK, not OpenAI Inc. API by default) ---
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")  # alias; use HF token when API_BASE_URL is HF router
    or os.getenv("API_KEY")
    or ""
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

TASK_NAME = os.getenv("VIDEO_ENCODE_TASK", "video_encode")
BENCHMARK = os.getenv("VIDEO_ENCODE_BENCHMARK", "video_encode")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "0.3"))
# Completion cap for the assistant message (JSON only). Raise if the provider truncates JSON.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4024"))
# Default off: ``response_format: json_object`` asks for a bare JSON object only — incompatible with
# delimiter lines ``<<<VIDEO_ENCODE_ACTION_JSON>>>``. Set INFERENCE_JSON_OBJECT_MODE=1 only if needed.
INFERENCE_JSON_OBJECT_MODE = os.getenv("INFERENCE_JSON_OBJECT_MODE", "0").strip().lower() not in (
    "0",
    "false",
    "no",
)
# Qwen3-style APIs: pass enable_thinking=false via extra_body (HF router / vLLM / many OpenAI-compat servers).
INFERENCE_DISABLE_THINKING = os.getenv("INFERENCE_DISABLE_THINKING", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.85"))

# Normalize score: assume typical |reward| per step ≤ this for rough [0,1] scaling
_MAX_REWARD_ABS_PER_STEP = float(os.getenv("VIDEO_ENCODE_SCORE_MAX_ABS_REWARD_PER_STEP", "3.0"))

# Default hosted Space (hyphens in hostname; underscores break TLS hostname match)
_DEFAULT_HF_SPACE_BASE_URL = os.getenv(
    "VIDEO_ENCODE_DEFAULT_HF_URL",
    "https://keshav142-video-encode.hf.space",
)

_WS_MESSAGE_TIMEOUT_SEC = float(os.getenv("VIDEO_ENCODE_WS_MESSAGE_TIMEOUT_SEC", "600"))

_PRINT_USER_PROMPT = os.getenv("VIDEO_ENCODE_PRINT_USER_PROMPT", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)


class ExtendedReadyLocalDockerProvider(LocalDockerProvider):
    """Longer /health wait for slow HF images."""

    def __init__(self, ready_timeout_s: float = 600.0) -> None:
        super().__init__()
        self._ready_timeout_s = float(ready_timeout_s)

    def wait_for_ready(self, base_url: str, timeout_s: float | None = None) -> None:
        t = self._ready_timeout_s if timeout_s is None else float(timeout_s)
        return super().wait_for_ready(base_url, timeout_s=t)


def _base_url_from_docker_container(container: str) -> str:
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
    if "]:" in line:
        host_port = int(line.split("]:")[1])
    else:
        host_port = int(line.rsplit(":", 1)[1])
    return f"http://127.0.0.1:{host_port}"


def _set_message_timeout(env: VideoEncodeEnv, sec: float) -> None:
    object.__setattr__(env, "_message_timeout", float(sec))


SYSTEM_PROMPT_BODY = textwrap.dedent(
    """
    You control a video encoding RL environment. Each step you choose libx264 parameters for one
    short segment of a reference video. The server extracts a clip, encodes it, measures VMAF/SSIM
    vs the reference, and returns a reward (higher is better when encoding succeeds).

    Schema for the JSON object you output (see OUTPUT CONTRACT below for how to emit it):
      {
        "crf": <integer 0-51>,
        "preset": "<string>",
        "video_index": <integer or null>
      }

    Fields:
      - crf: Constant Rate Factor; lower usually means higher quality and larger size (typical 18-28).
      - preset: libx264 speed preset. Allowed values include: ultrafast, superfast, veryfast, faster,
        fast, medium, slow, slower, veryslow.
      - video_index: Which loaded source video to use (0 .. num_videos-1), or null to keep the
        episode default from reset.

    Use the observation and history deliberately:
      - Read vmaf_score, ssim_score, bitrate_kbps, encoding_time_sec when present. If VMAF is high
        but bitrate is high, try a higher crf (e.g. +2 to +4) or a faster preset to seek efficiency.
        If VMAF is low, try lower crf or slower preset. If encoding_time_sec is large, consider a
        faster preset or slightly higher crf.
      - Use encode_step_index and segment_index: early steps in a segment can explore a wider range
        of crf/preset; later steps can refine around promising settings.
      - Read "Recent history": it lists past actions and rewards. Do NOT blindly repeat the same
        (crf, preset) as the immediately previous step unless you justify it from metrics (e.g. still
        tuning). When history already shows identical (crf, preset) multiple times with similar reward,
        you MUST change at least one of crf or preset (try e.g. crf±2..5, or preset one notch
        faster/slower) to explore the space.
      - If num_videos > 1, consider switching video_index when comparing sources; if num_videos is 1,
        use video_index 0 or null.

    Observation fields you will see in the user message (JSON):
      - echoed_message: Human-readable status or error from the server.
      - num_videos: Count of video files available.
      - current_video_index: Active source index after the last reset.
      - reset_index: How many resets so far in this session.
      - segment_index: Which timeline segment along the file (advances after K encodes per segment).
      - encode_step_index: Sub-step within the current segment (0 .. steps_per_segment-1).
      - steps_per_segment: K encodes per segment before the timeline advances.
      - prev_segment_predictions: Summary of the last completed segment (VMAF/SSIM means, etc.) or null.
      - whole_video_analysis: Optional analysis dict for the source file (duration, complexity metrics).
      - vmaf_score, ssim_score, bitrate_kbps, encoding_time_sec: Metrics from the last successful encode
        when present; may be null.
      - done: Episode terminal flag from the client (usually false until server says so).

    Goal: explore diverse (crf, preset) settings informed by metrics and history, and improve
    quality-efficiency tradeoffs and reward.
    """
).strip()


# Parser looks for JSON only between these exact lines (verbatim).
ACTION_JSON_START = "<<<VIDEO_ENCODE_ACTION_JSON>>>"
ACTION_JSON_END = "<<<END_VIDEO_ENCODE_ACTION_JSON>>>"


def _mandatory_delimiter_instructions(max_tokens: int) -> str:
    """Placed first in the system message so models see it before policy."""
    ex = '{"crf":23,"preset":"medium","video_index":0}'
    return textwrap.dedent(
        f"""
        ================================================================================
        CRITICAL — READ FIRST (the client parses ONLY this; no markers = failed step)
        ================================================================================
        Your reply MUST include these two lines EXACTLY as written (same characters, angle brackets,
        spelling, UPPERCASE). Copy-paste them; do not paraphrase, rename, or wrap in markdown.

        Line A (exactly): {ACTION_JSON_START}
        Line B (exactly one JSON object, compact, single line): {ex}
        Line C (exactly): {ACTION_JSON_END}

        Rules:
        1. Optional reasoning may appear ONLY above Line A. Do not put anything after Line C.
        2. Between Line A and Line C there must be ONLY the JSON line (Line B). No blank lines inside
           the block unless your JSON is a single line (preferred). No ``` fences.
        3. INVALID and will be rejected: output that is only JSON with no A/B/C lines; "Final answer:"
           without A/B/C; markdown code blocks; changing "VIDEO_ENCODE" or the number of > symbols.
        4. The last three non-empty lines of your message MUST be Line A, then Line B, then Line C
           (END line last). If you run out of tokens before Line C, the step fails — keep reasoning short.
        5. Whole-message token limit is about {max_tokens} tokens; prioritize finishing Lines A–C.
        6. Prefer putting the full reply (including Lines A–C) in the assistant ``content`` field.
           Some APIs leave ``content`` empty and return text in ``reasoning``; the client reads that
           when ``content`` is empty. Ensure Lines A–C appear in whichever field carries the reply.

        Example of a valid minimal reply (structure only; choose your own crf/preset):
        {ACTION_JSON_START}
        {ex}
        {ACTION_JSON_END}
        ================================================================================
        """
    ).strip()


def build_system_prompt(max_tokens: int) -> str:
    """Delimiter contract first, then encoding policy."""
    return f"{_mandatory_delimiter_instructions(max_tokens)}\n\n{SYSTEM_PROMPT_BODY}"


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_env_endpoint(
    docker_image: Optional[str],
    base_url: Optional[str],
    repo_id: Optional[str],
) -> None:
    """Log where the Video Encode RL server is reached (HF Space URL, repo, local, or Docker)."""
    if docker_image:
        # print(f"[ENV] mode=docker_image image={docker_image}", flush=True)
        pass
    elif repo_id:
        # print(f"[ENV] mode=hf_repo repo_id={repo_id} (OpenEnv from_env)", flush=True)
        pass
    elif base_url:
        # print(f"[ENV] mode=http url={base_url}", flush=True)
        pass
    else:
        # print("[ENV] mode=unknown", flush=True)
        pass


def log_user_prompt(step: int, user_prompt: str) -> None:
    if not _PRINT_USER_PROMPT:
        return
    # print(f"[USER_PROMPT] step={step}\n{user_prompt}\n", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    step: int,
    observation_dict: dict[str, Any],
    last_reward: float,
    history: List[str],
    max_tokens: int,
) -> str:
    hist = "\n".join(history[-16:]) if history else "(none)"
    obs_json = json.dumps(observation_dict, indent=2, default=str)
    return textwrap.dedent(
        f"""
        --- REMINDER: END YOUR MESSAGE WITH THIS EXACT 3-LINE BLOCK (copy the marker lines) ---
        {ACTION_JSON_START}
        {{"crf":<int>,"preset":"<string>","video_index":<int or null>}}
        {ACTION_JSON_END}
        (Put optional reasoning above the first line only. Do not output bare JSON without these lines.)

        Step number: {step}
        Last step reward: {last_reward:.4f}

        Current observation (JSON):
        {obs_json}

        Recent history (each line is a prior step: action JSON and reward — use this to avoid
        repeating the same crf/preset pair without reason, and to pick a new exploratory setting):
        {hist}

        Instructions:
        1. Compare planned (crf, preset) to values appearing in Recent history; prefer a different
           pair when rewards have plateaued or you are exploring early encode_step_index slots.
        2. Ground your choice in the observation (VMAF, bitrate, encoding time, segment/step indices).
        3. Your final output MUST be the three-line block above with real numbers/strings — not a
           summary, not "JSON:" without the marker lines. Budget ~{max_tokens} tokens total.
        """
    ).strip()


_ACTION_DELIMITED = re.compile(
    re.escape(ACTION_JSON_START) + r"\s*([\s\S]*?)\s*" + re.escape(ACTION_JSON_END),
    re.IGNORECASE,
)

# Prefer parsing the action block from the last N lines (final answer at end of message).
ACTION_JSON_SEARCH_LAST_LINES = int(os.getenv("VIDEO_ENCODE_ACTION_PARSE_LAST_LINES", "7"))


def _last_n_lines_text(text: str, n: int) -> str:
    if n <= 0:
        return text
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def _last_delimited_match(text: str) -> Optional[re.Match[str]]:
    """Return the **last** START…END match (handles duplicate or stray earlier blocks)."""
    matches = list(_ACTION_DELIMITED.finditer(text))
    return matches[-1] if matches else None


def _parse_action_dict_from_text(text: str) -> dict[str, Any]:
    """Extract action JSON from between ``ACTION_JSON_START`` and ``ACTION_JSON_END``.

    Searches the **last** ``ACTION_JSON_SEARCH_LAST_LINES`` lines first (typical final block). If
    multiple blocks appear there, the **last** pair wins. If not found in the tail, searches the
    full message the same way. Whitespace around the JSON line is stripped before ``json.loads``.
    """
    blob = (text or "").strip()
    if not blob:
        raise ValueError("empty model output")

    tail = _last_n_lines_text(blob, ACTION_JSON_SEARCH_LAST_LINES)
    m = _last_delimited_match(tail)
    if not m:
        m = _last_delimited_match(blob)
    if not m:
        raise ValueError(
            f"missing delimiters {ACTION_JSON_START!r} ... {ACTION_JSON_END!r} around the action JSON"
        )

    inner = m.group(1).strip()
    if not inner:
        raise ValueError("empty JSON between action delimiters")

    data = json.loads(inner)
    if not isinstance(data, dict):
        raise ValueError("delimited payload is not a JSON object")
    if not any(k in data for k in ("crf", "preset", "video_index")):
        raise ValueError("JSON object must include at least one of crf, preset, video_index")
    return data


def parse_model_output_to_action(text: str, num_videos: int, task_id: str) -> VideoEncodeAction:
    """Parse LLM text into a VideoEncodeAction; fall back to safe defaults on failure."""
    data = _parse_action_dict_from_text(text)
    if not isinstance(data, dict):
        raise ValueError("model output JSON is not an object")

    crf = int(max(0, min(51, int(data.get("crf", 23)))))
    preset = str(data.get("preset", "medium")).strip() or "medium"
    vi = data.get("video_index", None)
    video_index: int | None
    if vi is None or vi == "null":
        video_index = None
    else:
        video_index = int(vi)
        if num_videos > 0:
            video_index = max(0, min(video_index, num_videos - 1))

    try:
        return VideoEncodeAction(crf=crf, preset=preset, video_index=video_index, task_id=task_id)
    except ValidationError:
        return fallback_action(num_videos)


def fallback_action(num_videos: int) -> VideoEncodeAction:
    vi: int | None = 0 if num_videos > 0 else None
    return VideoEncodeAction(crf=23, preset="medium", video_index=vi, task_id="easy")


def _normalize_assistant_text_field(raw: Any) -> str:
    """Turn ``content`` / similar into a single string (handles str or multimodal part lists)."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: list[str] = []
        for p in raw:
            if isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif p.get("type") == "text" and isinstance(p.get("content"), str):
                    parts.append(p["content"])
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts).strip()
    return ""


def _completion_message_text(message: Any) -> tuple[str, str]:
    """Return (text, source_field) for delimiter/JSON parsing.

    Order: ``content`` first (preferred final channel). If empty, try ``reasoning``,
    ``reasoning_content``, ``text`` — many Hugging Face / Qwen routers put the full assistant
    string in ``reasoning`` and leave ``content`` blank. Optional ``model_dump()`` keys as last resort.
    """
    if message is None:
        return "", ""
    #print(f"[DEBUG] Model message: {message!r}", flush=True)
    # 1) content (string or part list)
    c = _normalize_assistant_text_field(getattr(message, "content", None))
    if c:
        return c, "content"

    # 2) common alternate field names on the message object
    for attr in ("reasoning", "reasoning_content", "text"):
        t = _normalize_assistant_text_field(getattr(message, attr, None))
        if t:
            return t, attr

    # 3) pydantic / dict fallbacks
    dump_fn = getattr(message, "model_dump", None)
    if callable(dump_fn):
        try:
            d = dump_fn()
            if isinstance(d, dict):
                for key in ("content", "reasoning", "reasoning_content", "text"):
                    t = _normalize_assistant_text_field(d.get(key))
                    if t:
                        return t, key
        except Exception:
            pass

    return "", ""


def _merge_disable_thinking_extra_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge ``enable_thinking: false`` into ``extra_body`` for Qwen3-style routers."""
    if not INFERENCE_DISABLE_THINKING:
        return kwargs
    thinking = {"enable_thinking": False}
    extra = kwargs.get("extra_body")
    if isinstance(extra, dict):
        merged = {**extra, **thinking}
    else:
        merged = thinking
    return {**kwargs, "extra_body": merged}


def _chat_completion_create(client: OpenAI, **kwargs: Any) -> Any:
    """Call chat.completions.create. ``json_object`` mode is optional — it conflicts with delimiter markers."""
    kwargs = _merge_disable_thinking_extra_body(kwargs)
    if INFERENCE_JSON_OBJECT_MODE:
        try:
            return client.chat.completions.create(
                **kwargs,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # print(
            #     f"[DEBUG] response_format json_object failed ({exc!r}); retrying without it",
            #     flush=True,
            # )
            pass
    return client.chat.completions.create(**kwargs)


def get_model_action(
    client: OpenAI,
    step: int,
    observation_dict: dict[str, Any],
    last_reward: float,
    history: List[str],
    num_videos: int,
    task_id: int,
) -> VideoEncodeAction:
    user_prompt = build_user_prompt(
        step, observation_dict, last_reward, history, max_tokens=MAX_TOKENS
    )
    #log_user_prompt(step, user_prompt)
    try:
        completion = _chat_completion_create(
            client,
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": build_system_prompt(MAX_TOKENS)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        choice = completion.choices[0]
        msg = choice.message
        text, text_source = _completion_message_text(msg)
        text = text.strip()
        fr = getattr(choice, "finish_reason", None)
        # print(
        #     f"[DEBUG] Assistant text: source={text_source!r} chars={len(text)} finish_reason={fr!r}",
        #     flush=True,
        # )
        if not text:
            raise ValueError(
                "empty assistant message: no text in content, reasoning, or other known fields"
            )
        # if fr == "length":
        #     print(
        #         "[DEBUG] Stopped at max_tokens — JSON may be truncated. Increase MAX_TOKENS or shorten JSON.",
        #         flush=True,
        #     )
        # if len(text) > 500:
        #     print(f"[DEBUG] Model text (head): {text}...", flush=True)
        # else:
        #     print(f"[DEBUG] Model text: {text!r}", flush=True)
        return parse_model_output_to_action(text, num_videos, task_id)
    except Exception as exc:
        # print(f"[DEBUG] Model request or parse failed: {exc}", flush=True)
        # print(
        #     "[DEBUG] Using fallback action: crf=23, preset=medium, video_index=0 "
        #     "(or null when no videos). Fix MODEL_NAME/API_BASE_URL/HF_TOKEN or model output.",
        #     flush=True,
        # )
        return fallback_action(num_videos)


def _normalize_hf_space_url(url: str) -> str:
    """HF Spaces hostnames use hyphens; underscores in the subdomain break TLS (hostname mismatch)."""
    u = url.strip()
    if not u or "://" not in u:
        return u
    from urllib.parse import urlparse, urlunparse

    p = urlparse(u)
    h = p.hostname
    if not h or not h.endswith(".hf.space") or "_" not in h:
        return u
    nh = h.replace("_", "-")
    new_netloc = p.netloc.replace(h, nh, 1)
    out = urlunparse(p._replace(netloc=new_netloc)).rstrip("/")
    if out != u.rstrip("/"):
        pass
        # print(f"[DEBUG] Normalized HF Space URL (use hyphens, not underscores): {u!r} -> {out!r}", flush=True)
    return out


def _resolve_base_url(args: argparse.Namespace) -> tuple[Optional[str], Optional[str]]:
    """Returns (base_url, repo_id) for env creation."""
    repo_id = (args.repo_id or os.environ.get("VIDEO_ENCODE_REPO_ID") or "").strip() or None
    docker_container = (args.docker_container or "").strip() or None

    if repo_id and (args.base_url or args.local or docker_container):
        raise ValueError("use --repo-id alone, or --base-url / --local / --docker-container")

    if docker_container and (args.base_url or args.local):
        raise ValueError("use either --docker-container or --base-url / --local")

    if repo_id:
        return None, repo_id
    if docker_container:
        return _base_url_from_docker_container(docker_container), None
    if args.base_url:
        return _normalize_hf_space_url(args.base_url), None
    if args.local:
        return "http://127.0.0.1:8000", None
    base = (
        os.environ.get("VIDEO_ENCODE_BASE_URL", "").strip()
        or os.environ.get("VIDEO_ENCODE_HF_URL", "").strip()
        or _DEFAULT_HF_SPACE_BASE_URL
    )
    return _normalize_hf_space_url(base), None


async def _create_env(
    base_url: Optional[str],
    repo_id: Optional[str],
    use_docker: bool,
    from_env_ready_timeout_s: float,
    docker_image: Optional[str],
) -> VideoEncodeEnv:
    if docker_image:
        provider = ExtendedReadyLocalDockerProvider(ready_timeout_s=from_env_ready_timeout_s)
        env = await VideoEncodeEnv.from_docker_image(docker_image, provider=provider)
        _set_message_timeout(env, _WS_MESSAGE_TIMEOUT_SEC)
        return env
    if repo_id:
        if use_docker:
            provider = ExtendedReadyLocalDockerProvider(ready_timeout_s=from_env_ready_timeout_s)
            env = await VideoEncodeEnv.from_env(repo_id, use_docker=True, provider=provider)
        else:
            env = await VideoEncodeEnv.from_env(repo_id, use_docker=False)
        _set_message_timeout(env, _WS_MESSAGE_TIMEOUT_SEC)
        return env
    assert base_url is not None
    env = VideoEncodeEnv(
        base_url=base_url,
        message_timeout_s=_WS_MESSAGE_TIMEOUT_SEC,
    )
    await env.connect()
    return env


def _action_to_log_str(action: VideoEncodeAction) -> str:
    return json.dumps(action.model_dump(), separators=(",", ":"), default=str)


def _history_line(step: int, action: VideoEncodeAction, reward: float) -> str:
    vi = action.video_index
    vi_s = "null" if vi is None else str(vi)
    return (
        f"step {step} | crf={action.crf} preset={action.preset} video_index={vi_s} "
        f"| reward={reward:.4f}"
    )


def _observation_error(obs: Any) -> Optional[str]:
    md = getattr(obs, "metadata", None) or {}
    if isinstance(md, dict) and md.get("error"):
        return str(md["error"])
    return None


async def run_inference(args: argparse.Namespace) -> None:
    if not API_KEY:
        # print(
        #     "Warning: Set HF_TOKEN (or OPENAI_API_KEY) to your Hugging Face token for "
        #     "google/flan-t5-small on router.huggingface.co — requests may fail without it.",
        #     file=sys.stderr,
        # )
        pass

    # HF router accepts Bearer token; placeholder only for local smoke without network.
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "hf")

    docker_image = (LOCAL_IMAGE_NAME or "").strip() or None
    if docker_image:
        base_url, repo_id = None, None
    else:
        base_url, repo_id = _resolve_base_url(args)

    ready_raw = os.environ.get("VIDEO_ENCODE_FROM_ENV_READY_TIMEOUT", "600")
    try:
        from_env_ready = float(args.ready_timeout) if args.ready_timeout is not None else float(ready_raw)
    except ValueError:
        from_env_ready = 600.0

    env: VideoEncodeEnv | None = None
    max_steps = max(0, args.max_steps)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=TASK_NAME, env_name=BENCHMARK, model=MODEL_NAME)
    log_env_endpoint(docker_image, base_url, repo_id)

    try:
        env = await _create_env(
            base_url=base_url,
            repo_id=repo_id,
            use_docker=not args.no_docker,
            from_env_ready_timeout_s=from_env_ready,
            docker_image=docker_image,
        )
        result = await env.reset()
        obs = result.observation
        last_reward = float(result.reward or 0.0)

        episode_policy = {
            "easy": [0.0,0.5],
            "medium": [0.5,0.8],
            "hard": [0.8,1.0]
        }


        for step in range(1, max_steps + 1):
            percent_idx = step / (max_steps+1)
            task_id = "easy"
            for task, pc in episode_policy.items():
                if pc[0] <= percent_idx <= pc[1]:
                    task_id = task
                    break

            
            if getattr(result, "done", False):
                break

            obs_dict = obs.model_dump()
            # print(obs_dict)
            action = get_model_action(
                client,
                step,
                obs_dict,
                last_reward,
                history,
                obs.num_videos,
                task_id
            )
            action_str = _action_to_log_str(action)

            result = await env.step(action)
            obs = result.observation
            reward = float(result.reward if result.reward is not None else 0.00001)
            done = bool(result.done)
            err = _observation_error(obs)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=err)

            history.append(_history_line(step, action, reward))

            if done:
                break

        mean_r = sum(rewards) / len(rewards)
        # Map roughly into [0,1] using typical scale (tunable via env)
        score = abs(mean_r)
        score = min(max(score, 0.0), 1.0)
        success = True

    except Exception as e:
        # print(f"[DEBUG] inference loop error: {e}", flush=True)
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                pass
                # print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> int:
    p = argparse.ArgumentParser(description="LLM inference client for Video Encode (OpenAI API + HF router).")
    p.add_argument(
        "--base-url",
        "--hf-url",
        dest="base_url",
        default=None,
        metavar="URL",
        help=(
            "Video Encode environment server URL (e.g. https://user-my-space.hf.space). "
            "Use hyphens in the subdomain, not underscores (TLS). "
            "If unset, use env VIDEO_ENCODE_HF_URL / VIDEO_ENCODE_BASE_URL or VIDEO_ENCODE_DEFAULT_HF_URL."
        ),
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Use http://127.0.0.1:8000 for the env (local server). Omit this to use a HF Space URL.",
    )
    p.add_argument("--docker-container", default=None, metavar="ID_OR_NAME")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--no-docker", action="store_true")
    p.add_argument("--ready-timeout", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = p.parse_args()

    try:
        asyncio.run(run_inference(args))
    except ValueError as e:
        # print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
