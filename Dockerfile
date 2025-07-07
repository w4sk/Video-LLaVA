FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential git ninja-build python3.10 python3.10-distutils python3-pip \
    python3-gi python3-gi-cairo gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-plugins-base-apps \
    python3-opencv \ 
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

WORKDIR /app

COPY pyproject.toml .

RUN pip install --upgrade pip

RUN pip install -e . && pip install -e ".[train]"
RUN pip install python-dotenv numpy flash-attn==2.2.0 --no-build-isolation
RUN pip install decord git+https://github.com/facebookresearch/pytorchvideo.git@28fe037d212663c6a24f373b94cc5d478c8c1a1d
