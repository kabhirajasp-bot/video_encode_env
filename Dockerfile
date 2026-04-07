# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Multi-stage build using openenv-base
# This Dockerfile is flexible and works for both:
# - In-repo environments (with local OpenEnv sources)
# - Standalone environments (with openenv from PyPI/Git)
# The build script (openenv build) handles context detection and sets appropriate build args.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Ensure git is available (required for installing dependencies from VCS)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Build argument to control whether we're building standalone or in-repo
ARG BUILD_MODE=in-repo
ARG ENV_NAME=video_encode

# Copy environment code (always at root of build context)
COPY . /app/env

COPY ../pyproject.toml /app/env/pyproject.toml

# For in-repo builds, openenv is already vendored in the build context
# For standalone builds, openenv will be installed via pyproject.toml
WORKDIR /app/env

# Ensure uv is available (for local builds where base image lacks it)
RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi
    
# Install dependencies using uv sync
# If uv.lock exists, use it; otherwise resolve on the fly
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

# Normalize uv path so the runtime stage can COPY it (base image may ship uv outside /usr/local/bin)
RUN if command -v uv >/dev/null 2>&1 && [ ! -x /usr/local/bin/uv ]; then \
        cp "$(command -v uv)" /usr/local/bin/uv; \
    fi

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Match local dev: `uv run server` (pyproject [project.scripts] server = video_encode.server.app:main)
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv
COPY --from=builder /usr/local/bin/uvx /usr/local/bin/uvx

WORKDIR /app/env

ENV HOST=0.0.0.0
ENV PORT=8000
# HF / Docker: load clips from the repo’s videos/ directory (see videos/README.md)
ENV VIDEO_ENCODE_VIDEOS_DIR=/app/env/videos

# curl (healthcheck) + ffmpeg/ffprobe (encode + VMAF)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/* && \
    ARCH="$(uname -m)" && \
    case "$ARCH" in \
        x86_64)   FFARCH=amd64 ;; \
        aarch64)  FFARCH=arm64 ;; \
        *) echo "unsupported architecture for static ffmpeg: $ARCH" >&2; exit 1 ;; \
    esac && \
    curl -fsSL "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${FFARCH}-static.tar.xz" \
        | tar xJ -C /tmp && \
    FDIR="$(find /tmp -maxdepth 1 -type d -name 'ffmpeg-*-'"${FFARCH}"'-static' | head -1)" && \
    test -n "$FDIR" && test -x "$FDIR/ffmpeg" && test -x "$FDIR/ffprobe" && \
    cp "$FDIR/ffmpeg" "$FDIR/ffprobe" /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf "$FDIR" && \
    ffmpeg -version | head -1 && \
    ffmpeg -filters 2>&1 | grep -q libvmaf



HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -sf "http://127.0.0.1:${PORT:-8000}/health" || exit 1

# Same as: cd project && uv run server
ENV ENABLE_WEB_INTERFACE=true
CMD ["sh", "-c", "cd /app/env && exec uv run server"]
