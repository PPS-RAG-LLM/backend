# Install
```
conda create -n niq python=3.13

conda activate niq

pip install uv

uv pip install -r requirements.txt
```

``` python
# .env
RDB_USER=postgres
RDB_PASSWORD=your_password
RDB_NAME=your_db_name

SSO_SECRET_KEY=NIQ_SOLUTION_SECRET_KEY
```


```
# milvus vector db 
docker compose -f docker-compose-milvus.yaml up -d

# postgreSQL 
docker compose -f docker-compose-postgresql.yaml up -d
```

```
python main.py
```



## Attu 컨테이너 시작 / 중지
milvus에 저장된 데이터를 보기 위해 Attu를 docker에 띄워접속
```
# UI를 보기 위해 

# 시작
## 호스트 번호
docker run -d -p 8000:3000 -e MILVUS_URL=172.17.0.1:19530 zilliz/attu:v2.4.0

# 중지
docker stop <컨테이너 ID>
```

### Attu 접속 방법


# ssh Port Forwarding
```
# 기존 터미널 server 실행
python main.py

# 터미널 새창(`config.yaml` 포트번호 참고)
ssh -L 3007:localhost:3007 elice_nipa
```


## Docker Install

### 1. Docker 설치 스크립트 다운로드 및 실행
```
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

### 2. 현재 사용자를 docker 그룹에 추가 (중요)
이 명령어를 입력해야 매번 sudo docker ...라고 치지 않고 그냥 docker ...로 쓸 수 있습니다.
```
sudo usermod -aG docker $USER
```

### 3. 적용을 위해 재접속
위 설정을 적용하려면 SSH 연결을 끊었다가 다시 접속

### 4. 재접속 후 확인
```
docker --version
docker compose version
```




# ERD 생성
```
pip install eralchemy
pip install SQLAlchemy
eralchemy -i sqlite:///my_old_sqlite.db -o my_new_er_diagram.pdf
```
```
python3 -m pip install pymilvus==2.6.0b0
pip install "pymilvus[model]"
pip install pymilvus
pip install \
    fastapi \
    streamlit \
    pymupdf \
    frontend \
    sentence-transformers \
    pandas \
    pyarrow \
    dill \
    aiohttp \
    numpy \
    accelerate \
    --upgrade --force-reinstall
```
