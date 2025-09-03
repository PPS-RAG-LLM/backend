
# utils/llms/adapters/qwen.py
from config import config
from utils.llms.registry import register, Streamer

from repository.users.llm_models import get_llm_model_by_provider_and_name
from utils import logger
from errors import NotFoundError
import os

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
        # Fallback: if DB missing, try default storage path
        storage_root = os.getenv("STORAGE_MODEL_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "storage", "model"))
        # Normalize to absolute path
        storage_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "storage", "model")) if not os.path.isabs(storage_root) else storage_root
        default_path = os.path.join(storage_root, model_key)
        logger.info(f"DB miss, trying filesystem fallback: {default_path}")
        local_path = default_path
    else:
        local_path = model_info.get("model_path")
    logger.info(f"hf_factory: {model_key}, {local_path}")

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