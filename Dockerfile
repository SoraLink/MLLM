FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python-is-python3 \
    git wget curl unzip vim ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# 安装 PyTorch（官方推荐轮子源）
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他包
RUN pip install transformers opencv-python gradio jupyterlab

WORKDIR /workspace
CMD ["/bin/bash"]