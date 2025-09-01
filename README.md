# Install
```
conda create -n coreIQ python=3.13

conda activate coreIQ

pip install uv

uv pip install -r requirements.txt
```

```
python main.py
```


# ssh Port Forwarding
```
# 기존 터미널 server 실행
python main.py

# 터미널 새창(`config.yaml` 포트번호 참고)
ssh -L 3007:localhost:3007 bai-vscode
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
