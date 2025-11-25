# Python 3.13 베이스 이미지 사용
# 공식 Python 이미지를 사용하여 Python 3.13 환경을 구성합니다
# CUDA 지원이 필요한 경우 nvidia/cuda 베이스 이미지를 사용할 수도 있습니다
FROM python:3.13-slim

# 작업 디렉토리 설정
# 모든 작업이 이 디렉토리에서 수행됩니다
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 빌드 도구 설치
# uv와 일부 Python 패키지 컴파일에 필요한 도구들을 설치합니다
# coreutils는 paste 명령을 포함하며, LD_LIBRARY_PATH 조정에 필요합니다
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
# uv는 빠른 Python 패키지 관리자로, pip보다 빠르게 의존성을 설치할 수 있습니다
RUN pip install --no-cache-dir uv

# 프로젝트 의존성 파일 복사
# uv.lock과 pyproject.toml을 먼저 복사하여 Docker 레이어 캐싱을 최적화합니다
# 이렇게 하면 코드가 변경되어도 의존성이 변경되지 않으면 캐시를 재사용할 수 있습니다
COPY uv.lock pyproject.toml ./

# uv를 사용하여 의존성 설치
# uv sync는 uv.lock 파일을 기반으로 정확한 버전의 패키지들을 설치합니다
# --frozen 옵션은 uv.lock 파일을 정확히 따르도록 보장합니다
RUN uv sync --frozen

# 애플리케이션 코드 복사
# 나머지 애플리케이션 코드를 복사합니다
COPY . .

# uv 환경을 활성화하고 Python 경로 설정
# uv로 설치한 패키지들이 사용 가능하도록 환경 변수를 설정합니다
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# GPU 설정 (기본값: GPU 1번 사용)
# CUDA_VISIBLE_DEVICES를 설정하여 사용할 GPU를 지정합니다
# docker run 시 -e CUDA_VISIBLE_DEVICES=0 등으로 변경 가능합니다
ENV CUDA_VISIBLE_DEVICES=1

# 포트 노출
# FastAPI 애플리케이션에서 사용할 포트를 노출합니다
EXPOSE 3007

# 애플리케이션 실행
# LD_LIBRARY_PATH 조정 후 uvicorn을 실행합니다
# torch 라이브러리 경로를 제거하여 라이브러리 충돌을 방지합니다
CMD ["sh", "-c", "export LD_LIBRARY_PATH=$(echo \"$LD_LIBRARY_PATH\" | tr ':' '\\n' | grep -v '/usr/local/lib/python3.10/dist-packages/torch/lib' | paste -sd:) && exec uvicorn main:app --host 0.0.0.0 --port 3007"]

