ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

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

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# keep the build context clean
WORKDIR /build          
COPY . /build

# Install core dependencies needed for FlashInfer
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch filelock packaging requests numpy ninja

# Install FlashInfer from source
ARG FLASHINFER_GIT_REPO="https://github.com/flashinfer-ai/flashinfer.git"
ARG FLASHINFER_GIT_REF="v0.2.8"
RUN --mount=type=cache,target=/root/.cache/uv bash -xe <<'BASH'
    set -euo pipefail
    . /etc/environment
    
    # Clone FlashInfer repository
    echo "📦 Cloning FlashInfer ${FLASHINFER_GIT_REF}..."
    git clone --depth 1 --recursive --shallow-submodules \
        --branch ${FLASHINFER_GIT_REF} \
        ${FLASHINFER_GIT_REPO} flashinfer || exit 1
    
    # Determine CUDA architectures based on version
    if [[ "${CUDA_VERSION}" == 11.* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9"
    elif [[ "${CUDA_VERSION}" == 12.[0-7]* ]]; then
        export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a"
    else
        export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.9 9.0a 10.0a 12.0"
    fi
    echo "🏗️  Building for CUDA arches: ${TORCH_CUDA_ARCH_LIST}"
    
    # Build and install FlashInfer
    cd flashinfer
    
    # Set CUDA environment
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    echo "⚙️  Skipping AOT kernels (will be built on first use)..."
    # Note: AOT kernels require nvshmem which isn't available
    # They will be JIT compiled on first use instead
    
    echo "📦 Installing FlashInfer..."
    uv pip install --system --no-build-isolation --force-reinstall --no-deps . || exit 1
    
    cd ..
    rm -rf flashinfer
    echo "✅ FlashInfer installed successfully"
BASH


# Needed to resolve setuptools dependencies
ENV UV_INDEX_STRATEGY="unsafe-best-match"
RUN uv pip install --system --no-cache ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --system --no-cache ".[bench]"

RUN playwright install-deps
RUN playwright install chromium

RUN python3 -m olmocr.pipeline --help