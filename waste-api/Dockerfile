# waste-api Docker 이미지 — Hugging Face Spaces / Render / Fly.io 등 어디서나 동작
# Hugging Face Spaces 가 기대하는 포트: 7860 (PORT 환경변수로 주입됨)

FROM python:3.11-slim

# 시스템 라이브러리 (Pillow, onnxruntime 종속)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 (Docker layer 캐싱)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 서빙 모델 — git(LFS) 대신 HF Hub `ethanDev92/waste-models/serving/` 에서 빌드 시 다운로드.
# (2026-08-30 모노레포 git 단일화: Space 저장소에서 LFS 모델 제거. 모델 갱신은
#  waste-classifier publish 스크립트가 HF Hub 에 올리고, 여기서는 재빌드만 하면 됨.)
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('ethanDev92/waste-models', allow_patterns=['serving/**'], local_dir='/tmp/wm')" \
    && mv /tmp/wm/serving /app/models && rm -rf /tmp/wm

# 앱 코드
COPY src ./src
COPY main.py .
COPY design ./design

# HF Spaces 기본 포트
ENV PORT=7860
EXPOSE 7860

# 모델 경로 명시 (config._resolve_model_path 가 우선 인식)
ENV WASTE_API_MODEL_PATH=/app/models/classifier.onnx

# SUPABASE_URL / SUPABASE_KEY 는 HF Spaces Secrets 로 주입
# (Settings → Variables and secrets 에서 추가)

CMD ["python", "main.py", "--host", "0.0.0.0"]
