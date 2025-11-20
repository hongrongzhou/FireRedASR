FROM python:3.9-slim-bullseye

ENV TZ=Asia/Shanghai
ENV MODEL_PATH=/workspace/FireRedASR/models/small
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    git-lfs \
    ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git config --global url."https://github.com/".insteadOf git@github.com: \
    && git lfs install \
    && git clone https://github.com/hongrongzhou/FireRedASR.git --depth 1 \
    && cd FireRedASR \
    && git lfs pull \
    && pip3 install --upgrade pip \
    && pip3 install -r requirements.txt torch torchaudio --no-cache-dir

EXPOSE 8080

WORKDIR /workspace/FireRedASR
CMD ["sh", "-c", "python3 fire_red_asr.py --model_path $MODEL_PATH --port 8080"]
