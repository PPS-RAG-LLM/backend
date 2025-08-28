
# utils/llms/adapters/qwen.py
from config import config
from utils.llms.registry import register, Streamer
import os
from pathlib import Path

from repository.users.llm_models import get_llm_model_by_provider_and_name
from utils import logger
from errors import NotFoundError

logger = logger(__name__)

### Huggingface

class _Wrap:
	def __init__(self, fn): self.fn = fn
	def stream(self, messages, **kw): return self.fn(messages, **kw)


@register("huggingface")
def hf_factory(model_key: str) -> Streamer:
    from utils.llms.huggingface import qwen_7b, qwen_vl_7b, gpt_oss_20b
    # 데이터베이스에서 모델 정보 조회
    logger.info(f"hf_factory: {model_key}")
    model_info = get_llm_model_by_provider_and_name("huggingface", model_key)
    logger.info(f"model_info: {model_info}")
    if not model_info:
        raise NotFoundError(f"지원하지 않는 HF 모델: {model_key}")

    # 오프라인 환경을 위해 DB의 model_path가 없으면 storage/model/<name>로 해석
    try:
        backend_root = Path(__file__).resolve().parents[3]  # .../backend
        storage_model_root = os.path.join(str(backend_root), "storage", "model")
    except Exception:
        # 최악의 경우 현재 위치 기준 상대 경로 시도
        storage_model_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "storage", "model"))

    name = model_info.get("name") or model_key
    # DB의 model_path 우선, 없으면 storage/model/<name>
    local_path = model_info.get("model_path") or os.path.join(storage_model_root, name)
    # 상대 경로면 storage/model 하위로 보정
    if local_path and not os.path.isabs(local_path):
        cand = os.path.join(storage_model_root, local_path)
        if os.path.isfile(os.path.join(cand, "config.json")):
            local_path = cand
    logger.info(f"hf_factory resolved path: {local_path}")

    # 모델 패밀리에 따라 적절한 Streamer 생성
    if model_key == "qwen2.5-7b-instruct":
        return _Wrap(lambda messages, **kw: qwen_7b.stream_chat(messages, model_path=local_path, **kw))
    if model_key == "qwen2.5-vl-7b-instruct":
        return _Wrap(lambda messages, **kw: qwen_vl_7b.stream_chat(messages, model_path=local_path, **kw))
    if model_key.startswith("gpt_oss"):
        return _Wrap(lambda messages, **kw: gpt_oss_20b.stream_chat(messages, model_path=local_path, **kw))
    
    raise NotFoundError(f"지원하지 않는 huggingface 모델: {model_key}")

### OpenAI API

class OpenAIStreamer:
	def __init__(self, model: str):
		self.model = model
	def stream(self, messages, **kw):
		from utils.llms.openai.streamer import stream_chat as openai_stream
		return openai_stream(messages, model=self.model, **kw)

@register("openai")
def openai_factory(model_key: str) -> Streamer:
	return OpenAIStreamer(model_key)