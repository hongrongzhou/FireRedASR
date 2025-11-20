FROM registry.cn-hangzhou.aliyuncs.com/public/ubuntu:22.04

ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "deb http://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update -y --fix-missing && \
    apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-dev \
        ffmpeg libsndfile1 libopenblas-dev libgfortran5 \
        gcc g++ make git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir \
    --default-timeout=1200 --retries=5 \
    cn2an kaldiio kaldi_native_fbank numpy peft sentencepiece \
    torch torchaudio torchvision transformers \
    fastapi uvicorn python-multipart noisereduce soundfile

ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONPATH=/app:$PYTHONPATH
ENV PATH=/app/fireredasr:/app/fireredasr/utils:$PATH

RUN mkdir -p pretrained_models && \
    git clone https://hf-mirror.com/FireRedTeam/FireRedASR-AED-small pretrained_models/FireRedASR-AED-small

RUN echo "from fastapi import FastAPI, UploadFile, File; import tempfile; import os; from fireredasr.models.fireredasr import FireRedAsr; app = FastAPI(); model = FireRedAsr.from_pretrained('aed', 'pretrained_models/FireRedASR-AED-small', {'use_gpu':0}); @app.post('/asr') async def asr(file: UploadFile = File(...)): with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp: tmp.write(await file.read()); tmp_path = tmp.name; try: results = model.transcribe(['tmp_uttid'], [tmp_path], {'beam_size':3, 'nbest':1}); return {'text': results[0]['text']} finally: os.remove(tmp_path); @app.get('/health') async def health(): return {'status': 'ok'}" > main.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
