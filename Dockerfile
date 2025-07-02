FROM python:3.11-slim

# 필수 시스템 패키지 설치 (onnx, torch 등 일부는 필요함)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Poetry 설치 (버전 고정)
ENV POETRY_VERSION=1.8.5
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=$POETRY_VERSION python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Docker 안에서 가상환경 생성 방지
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# 의존성 파일 복사
COPY pyproject.toml poetry.lock* ./

# 의존성 설치
RUN poetry install --no-interaction --no-ansi

# 앱 전체 복사
COPY . .

# 실행 (FastAPI 서버 실행)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
