ARG CUDA_VERSION=12.8.1
FROM --platform=linux/amd64 nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Needs to be repeated below the FROM, or else it's not picked up
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION=12.8.1

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
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv

# olmOCR Specific Installs - Install fonts BEFORE changing Python version
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends poppler-utils fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools ttf-mscorefonts-installer

# Now update Python alternatives
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install uv for faster pip installs
RUN --mount=type=cache,target=/root/.cache/uv python3 -m pip install uv

# Install some helper utilities for things like the benchmark
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    unzip

ENV PYTHONUNBUFFERED=1

# keep the build context clean
WORKDIR /build          
COPY . /build

# Install FlashInfer from source
ARG FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
# Keep this in sync with https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt
# We use `--force-reinstall --no-deps` to avoid issues with the existing FlashInfer wheel.
ARG FLASHINFER_GIT_REF="v0.2.11"
RUN --mount=type=cache,target=/root/.cache/uv bash - <<'BASH'
  . /etc/environment
    git clone --depth 1 --recursive --shallow-submodules \
        --branch ${FLASHINFER_GIT_REF} \
        ${FLASHINFER_GIT_REPO} flashinfer
    # Exclude CUDA arches for older versions (11.x and 12.0-12.7)
    # TODO: Update this to allow setting TORCH_CUDA_ARCH_LIST as a build arg.
    if [[ "${CUDA_VERSION}" == 11.* ]]; then
        FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
    elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
        FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
    else
        # CUDA 12.8+ supports 10.0a and 12.0
        FI_TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
    fi
    echo "🏗️  Building FlashInfer for arches: ${FI_TORCH_CUDA_ARCH_LIST}"
    # Needed to build AOT kernels
    pushd flashinfer
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            python3 -m flashinfer.aot
        TORCH_CUDA_ARCH_LIST="${FI_TORCH_CUDA_ARCH_LIST}" \
            uv pip install --system --no-build-isolation --force-reinstall --no-deps .
    popd
    rm -rf flashinfer
BASH

# Needed to resolve setuptools dependencies
ENV UV_INDEX_STRATEGY="unsafe-best-match"
RUN uv pip install --system --no-cache ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system --no-cache ".[bench]"

RUN playwright install-deps
RUN playwright install chromium

RUN python3 -m olmocr.pipeline --help