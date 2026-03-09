import os
import re
import yaml
from dotenv import load_dotenv
load_dotenv()

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')


def _resolve_env_vars(obj):
    """YAML 로드 후 모든 문자열 값에서 ${VAR}를 환경변수로 치환한다."""
    if isinstance(obj, str):
        return _ENV_VAR_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    return obj


def load_config(path="config.yaml"):
    with open(path, "r", encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    return _resolve_env_vars(raw)

config = load_config()

# 사용 예시
# print(config["qwen"]["model_path"])
# print(config["openai"]["api_key"])