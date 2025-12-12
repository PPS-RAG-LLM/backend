# Docker 빌드 타임아웃 방지 가이드

## 문제
- 빌드 컨텍스트가 150GB+로 매우 큼
- 모델 파일 포함 시 빌드 시간이 오래 걸림
- 타임아웃 발생 가능

## 해결 방법

### 1. Docker Daemon 타임아웃 설정 (시스템 레벨)

#### 방법 A: Docker daemon.json 수정
```bash
sudo nano /etc/docker/daemon.json
```

다음 내용 추가:
```json
{
  "max-concurrent-downloads": 3,
  "max-concurrent-uploads": 5,
  "max-download-attempts": 5
}
```

Docker 재시작:
```bash
sudo systemctl restart docker
```

#### 방법 B: 환경 변수 설정
```bash
# 빌드 전에 실행
export DOCKER_BUILDKIT=1
export BUILDKIT_STEP_LOG_MAX_SIZE=50000000
export BUILDKIT_STEP_LOG_MAX_SPEED=10000000
```

### 2. 빌드 스크립트 사용 (권장)

```bash
# 빌드 스크립트 실행
./docker-build.sh

# 또는 타임아웃 시간 지정
BUILD_TIMEOUT=10800 ./docker-build.sh  # 3시간
```

### 3. 직접 빌드 시 옵션

```bash
# 네트워크 최적화 + 상세 로그
docker build \
    --network=host \
    --progress=plain \
    --no-cache \
    -t niq:v0.0.1 \
    -f Dockerfile \
    . 2>&1 | tee build.log
```

### 4. 모델을 별도로 관리 (선택사항)

모델이 너무 크면 볼륨 마운트 사용:

```bash
# 빌드 시 모델 제외
# .dockerignore에 storage/models/ 추가

# 실행 시 볼륨 마운트
docker run -d \
    -p 3007:3007 \
    -v /home/wzxcv123/NIQ/jo/backend/storage/models:/app/storage/models \
    -v /home/wzxcv123/NIQ/jo/backend/storage/models/embedding:/app/storage/models/embedding \
    --restart always \
    --name NIQ \
    niq:v0.0.1
```

### 5. 빌드 최적화 팁

1. **모델을 먼저 복사**: Dockerfile에서 모델을 별도 레이어로 분리 (이미 적용됨)
2. **불필요한 파일 제외**: .dockerignore 최적화 (이미 적용됨)
3. **멀티스테이지 빌드**: 필요시 고려 가능

### 6. 빌드 진행 상황 모니터링

```bash
# 별도 터미널에서 빌드 로그 확인
tail -f build.log

# 또는 Docker 빌드 진행 상황 확인
docker ps -a
```

## 현재 설정

- ✅ 모델 파일 포함 (storage/models/, storage/models/embedding/, storage/models/embedding-rerank)
- ✅ 불필요한 파일 제외 (.dockerignore 최적화)
- ✅ 모델을 별도 레이어로 분리 (캐싱 최적화)
- ✅ 빌드 스크립트 제공 (docker-build.sh)

## 예상 빌드 시간

- 모델 포함 시: 2-4시간 (네트워크 속도에 따라 다름)
- 모델 제외 시: 10-20분

## 문제 해결

빌드가 계속 실패하면:
1. 디스크 공간 확인: `df -h`
2. Docker 로그 확인: `docker system df`
3. 빌드 캐시 정리: `docker builder prune -a -f`
4. 모델을 볼륨 마운트로 변경 고려

