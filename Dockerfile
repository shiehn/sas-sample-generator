# Optional custom image for RunPod. Use only once you're past prototyping —
# the stock "PyTorch 2.x CUDA 12.x" template plus scripts/setup.sh is simpler
# while you're iterating on prompts.
#
# Build + push:
#   docker build -t YOUR_DOCKERHUB_USER/sas-sample-generator:latest .
#   docker push YOUR_DOCKERHUB_USER/sas-sample-generator:latest
#
# Then in RunPod, pick "Deploy a custom image" and paste the tag.
#
# Model weights are NOT baked in. They download on first run into the HF cache
# on the persistent volume — same as the stock-template flow.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3-pip \
      git curl ca-certificates openssh-client \
      libsndfile1 ffmpeg rclone zip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /opt/sas-sample-generator

COPY requirements.txt ./
RUN python -m pip install --upgrade pip wheel setuptools \
 && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 && pip install -r requirements.txt

COPY scripts ./scripts
COPY prompts ./prompts

# Outputs + HF cache live on the persistent volume mounted at /workspace.
ENV HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    SAS_OUTPUTS_DIR=/workspace/outputs

CMD ["bash"]
