FROM --platform=linux/amd64 nvidia/cuda:12.8.0-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update

RUN apt-get update && apt-get -y install python3-apt
RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections
RUN apt-get update -y && apt-get install -y poppler-utils ttf-mscorefonts-installer msttcorefonts fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    ca-certificates \
    build-essential \
    curl \
    wget \
    unzip

RUN rm -rf /var/lib/apt/lists/* \
    && unlink /usr/bin/python3 \
    && ln -s /usr/bin/python3.11 /usr/bin/python3 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && pip3 install -U pip   

RUN apt-get update && apt-get -y install python3.11-venv 
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

ENV PYTHONUNBUFFERED=1

WORKDIR /root
COPY pyproject.toml pyproject.toml
COPY olmocr/version.py olmocr/version.py

RUN /root/.local/bin/uv pip install --system --no-cache -e .
RUN /root/.local/bin/uv pip install --system --no-cache ".[gpu]" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
RUN /root/.local/bin/uv pip install --system --no-cache ".[bench]"
RUN playwright install-deps
RUN playwright install chromium
COPY olmocr olmocr
COPY scripts scripts

RUN python3 -m sglang.launch_server --help
RUN python3 -m olmocr.pipeline --help