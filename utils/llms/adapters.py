
# utils/llms/adapters/qwen.py
from config import config
from utils.llms.registry import register, Streamer
from repository.llm_models import get_llm_model_by_provider_and_name
from utils import logger
from errors import NotFoundError
import os
from typing import Dict, Optional

logger = logger(__name__)

### Huggingface

class _Wrap:
	def __init__(self, fn): self.fn = fn
	def stream(self, messages, **kw): return self.fn(messages, **kw)


_HF_STREAMER_CACHE: Dict[str, Streamer] = {}


def clear_hf_streamer_cache(model_key: Optional[str] = None) -> None:
    """테스트/관리 용도로 HF 스트리머 캐시를 비운다."""
    if model_key:
        _HF_STREAMER_CACHE.pop(model_key, None)
    else:
        _HF_STREAMER_CACHE.clear()


def _resolve_model_path(local_path: str) -> str:
    from pathlib import Path
    if not local_path:
        raise NotFoundError("DB에 model_path가 비어 있습니다.")
    
    base = Path(config.get("models_dir").get("llm_models_path"))

    if os.path.isabs(local_path):
        abs_path = Path(local_path)
    else:
        s = local_path.lstrip("./")
        if "storage/models/llm/" in s :
            suffix = s.split("storage/models/llm/", 1)[1].strip("/")
            abs_path = (base / suffix)
        else:
            abs_path = base / os.path.basename(s)
    abs_path = abs_path.resolve()
    logger.info(f"수정된 모델경로 확인: {abs_path}")
    
    return abs_path

@register("huggingface")
def hf_factory(model_key: str) -> Streamer:
    from utils.llms.huggingface import qwen, gpt_oss_20b, gemma3_27b
    from pathlib import Path

    if model_key in _HF_STREAMER_CACHE:
        logger.debug("hf_factory cache hit: %s", model_key)
        return _HF_STREAMER_CACHE[model_key]

    # # 데이터베이스에서 모델 정보 조회
    logger.info(f"hf_factory: {model_key}")
    model_info = get_llm_model_by_provider_and_name("huggingface", model_key)
    logger.info(f"model_info: {model_info}")

    # # 모델 경로 확인
    local_path = _resolve_model_path(model_info.get("model_path"))
    # logger.info(f"local_path: {local_path}")
    logger.info(f"hf_factory: {model_key}")

    if not os.path.isdir(local_path):
        raise NotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {local_path}")

    # 모델 패밀리에 따라 적절한 Streamer 생성
    streamer: Optional[Streamer] = None

    if model_key.startswith("Qwen3-8B") or model_key.startswith("Qwen3-14B"):
        logger.info("Alibaba | Qwen Model %s", model_key)
        streamer = _Wrap(lambda messages, **kw: qwen.stream_chat(messages, model_path=local_path, **kw))

    elif model_key.startswith("Qwen3-vl"):
        from utils.llms.huggingface import qwen_vl
        logger.info("Alibaba | Qwen Model Qwen3-vl")
        streamer = _Wrap(lambda messages, **kw: qwen_vl.stream_chat(messages, model_path=local_path, **kw))

    elif model_key.startswith("Gemma"):
        logger.info("Google | Gemma Model %s", model_key)
        streamer = _Wrap(lambda messages, **kw: gemma3_27b.stream_chat(messages, model_path=local_path, **kw))

    elif model_key.startswith("gpt_oss") or model_key.startswith("gpt-oss"):
        logger.info("gpt_oss_20b")
        streamer = _Wrap(lambda messages, **kw: gpt_oss_20b.stream_chat(messages, model_path=str(local_path), **kw))

    if streamer is None:
        logger.error("해당모델 이름으로 시작하는 로직이 없음. %s", model_key)
        raise NotFoundError(f"지원하지 않는 huggingface 모델: {model_key}")

    _HF_STREAMER_CACHE[model_key] = streamer
    return streamer


def preload_adapter_model(model_key: str) -> bool:
    """Preload a local HF model via shared loader so that adapter path can stream later.
    This avoids bitsandbytes/triton by using utils.model_load.load_hf_llm_model.
    Returns True on success, False otherwise.
    """
    try:
        model_info = get_llm_model_by_provider_and_name("huggingface", model_key)
        local_path = _resolve_model_path(model_info.get("model_path"))
        from utils import load_hf_llm_model
        load_hf_llm_model(str(local_path))
        try:
            logger.info(f"adapter preload ok: {model_key} -> {local_path}")
        except Exception:
            pass
        return True
    except Exception as e:
        try:
            logger.exception(f"adapter preload failed for {model_key}: {e}")
        except Exception:
            pass
        return False

### OpenAI API

class OpenAIStreamer:
    # 생성자에서 api_key를 받아서 저장해둠
    def __init__(self, default_model: str, api_key: str = None):
        self.default_model = default_model
        self.api_key = api_key  # <-- 저장!

    def stream(self, messages, **kw):
        from utils.llms.openai.streamer import stream_chat as openai_stream
        
        if "model" not in kw or not kw["model"]:
            kw["model"] = self.default_model
        
        # 생성자에서 받은 api_key를 호출 시점에 주입 (이미 kw에 있으면 덮어쓰지 않음)
        if self.api_key and "api_key" not in kw:
            kw["api_key"] = self.api_key
            
        return openai_stream(messages, **kw)

@register("openai")
def openai_factory(model_key: str, **kwargs) -> Streamer:
    api_key = kwargs.get("api_key")
    return OpenAIStreamer(default_model=model_key, api_key=api_key)