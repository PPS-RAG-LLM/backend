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
ssh -L 3005:localhost:3005 bai-vscode
```



# ERD 생성
```
pip install eralchemy
pip install SQLAlchemy
eralchemy -i sqlite:///my_old_sqlite.db -o my_new_er_diagram.pdf
```