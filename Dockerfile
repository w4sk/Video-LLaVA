FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential git ninja-build python3.11 python3.11-distutils python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip 

RUN pip install -e . && pip install -e ".[train]"
RUN pip install flash-attn==2.2.0 --no-build-isolation
RUN pip install decord opencv-python git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
