FROM amdp-registry.skala-ai.com/skala25a/skoro-ai-base:1.0.0
# FROM python:3.11-slim

# # 필수 시스템 패키지 설치 (onnx, torch 등 일부는 필요함)
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libglib2.0-0 \
#     libsm6 \
#     libxrender1 \
#     libxext6 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# WORKDIR /app

# # requirements.txt만 먼저 복사해서 설치 (캐시 활용)
# COPY requirements.txt .

# RUN pip install --no-cache-dir -r requirements.txt

# 앱 전체 복사
COPY . .

# 실행 (FastAPI 서버 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]