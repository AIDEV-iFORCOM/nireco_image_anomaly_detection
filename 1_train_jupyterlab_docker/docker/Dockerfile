# CUDA 12.2 + cuDNN8 公式ランタイム利用
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# -------- OS パッケージ --------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev git wget \
        libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------- Python ライブラリ --------
COPY docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python3 -m pip install --no-cache-dir \
    # JupyterLab
        jupyterlab \
        jupyter-server-proxy \
        jupyter-tensorboard-proxy && \
    jupyter server extension enable --sys-prefix jupyter_server_proxy

# -------- 作業ディレクトリ --------
RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]