# Install
```
conda create -n rag python=3.13

conda activate rag

pip install uv

uv pip install -r requirements.txt
```

```
uvicorn src.api.main:app --host 172.16.0.105 --port 8000
```
