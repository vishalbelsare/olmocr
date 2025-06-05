ARG CUDA_VERSION=12.6.0

FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.6.0

# Set environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# From original VLLM dockerfile https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile
# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo python3-apt \
    && for i in 1 2 3; do \
        add-apt-repository -y ppa:deadsnakes/ppa && break || \
        { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
    done \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv \
    python3 -m pip install uv

# olmOCR Specific Installs
# Install fonts with workaround for update-notifier issue
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends poppler-utils fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools && \
    # Temporarily fix the python symlink for the installer
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    apt-get install -y --no-install-recommends ttf-mscorefonts-installer && \
    # Restore our Python 3.12 symlink
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

# Install some helper utilities for things like the benchmark
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    unzip

ENV PYTHONUNBUFFERED=1

WORKDIR /root
COPY pyproject.toml pyproject.toml
COPY olmocr/version.py olmocr/version.py

# Needed to resolve setuptools dependencies
ENV UV_INDEX_STRATEGY="unsafe-best-match"

RUN uv pip install --system --no-cache-dir "sglang[all]>=0.4.6.post5"
RUN uv pip install --system --no-cache-dir -e .
RUN uv pip install --system --no-cache ".[bench]"
RUN playwright install-deps
RUN playwright install chromium
COPY olmocr olmocr
COPY scripts scripts


RUN python3 -m olmocr.pipeline --help