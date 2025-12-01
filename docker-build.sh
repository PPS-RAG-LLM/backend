#!/bin/bash
# Docker 빌드 스크립트 (타임아웃 방지)

# Docker 빌드 타임아웃 설정 (초 단위)
# 기본값은 없지만, 빌드 컨텍스트가 클 경우 시간이 오래 걸릴 수 있음
BUILD_TIMEOUT=${BUILD_TIMEOUT:-90000}  # 25시간 기본값

echo "=========================================="
echo "Docker 이미지 빌드 시작"
echo "타임아웃: ${BUILD_TIMEOUT}초 (25시간)"
echo "=========================================="

# Docker 빌드 실행
# --network=host: 네트워크 속도 향상
# --progress=plain: 상세한 빌드 로그
# --no-cache: 캐시 없이 빌드 (필요시 제거)
docker build \
    --network=host \
    --progress=plain \
    --no-cache \
    -t niq:v0.0.1 \
    -f Dockerfile \
    . 2>&1 | tee build.log

BUILD_EXIT_CODE=${PIPESTATUS[0]}

if [ $BUILD_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "✅ 빌드 성공!"
    echo "=========================================="
    echo "이미지 실행: docker run -d -p 3007:3007 --restart always --name NIQ niq:v0.0.1"
else
    echo "=========================================="
    echo "❌ 빌드 실패 (종료 코드: $BUILD_EXIT_CODE)"
    echo "=========================================="
    echo "빌드 로그 확인: tail -100 build.log"
    exit $BUILD_EXIT_CODE
fi

