# Use an NVIDIA base image with CUDA support

ARG CUDA_IMAGE="12.8.1-devel-ubuntu24.04"

FROM nvidia/cuda:${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.6
ARG TORCHVISION_VERSION=0.21
ARG TORCHAUDIO_VERSION=2.6
ARG CUDA_SHORT_VERSION=12.8

# Set working directory
WORKDIR /

RUN echo PYTHON_VERSION=${PYTHON_VERSION} \
    && echo CUDA_SHORT_VERSION=${CUDA_SHORT_VERSION} \
    && echo TORCH_VERSION=${TORCH_VERSION} \
    && echo TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    && echo TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}

# Setup timezone and install system dependencies
RUN 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select New_York' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt update && apt install wget git -y && apt clean

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniconda.sh \
    && ${CONDA_DIR}/bin/conda clean -afy

ENV PATH=${CONDA_DIR}/bin:$PATH

RUN conda init bash

RUN conda create -y -n nunchaku python=${PYTHON_VERSION} \
    && conda install -y -n nunchaku -c conda-forge gxx=11 gcc=11 \
    && conda clean -afy

SHELL ["conda", "run", "-n", "nunchaku", "/bin/bash", "-c"]

# Install building dependencies
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu124
RUN pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub comfy-cli

# Start building
RUN git clone https://github.com/mit-han-lab/nunchaku.git \
    && cd nunchaku \
    && git submodule init \
    && git submodule update \
    && NUNCHAKU_INSTALL_MODE=ALL python setup.py develop

RUN cd .. && git clone https://github.com/comfyanonymous/ComfyUI \
    && cd ComfyUI && pip install -r requirements.txt \
    && cd custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager \
    && git clone https://github.com/mit-han-lab/ComfyUI-nunchaku.git \
    && cd .. && mkdir -p user/default/workflows/ && cp -r custom_nodes/ComfyUI-nunchaku/workflows/ user/default/workflows/nunchaku_examples
