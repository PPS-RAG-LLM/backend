
# utils/llms/adapters/qwen.py
from config import config
from utils.llms.registry import register, Streamer
from repository.llm_models import get_llm_model_by_provider_and_name
from utils import logger
from errors import NotFoundError
import os

logger = logger(__name__)

### Huggingface

class _Wrap:
	def __init__(self, fn): self.fn = fn
	def stream(self, messages, **kw): return self.fn(messages, **kw)


def _resolve_model_path(local_path: str) -> str:
    from pathlib import Path
    if not local_path:
        raise NotFoundError("DB에 model_path가 비어 있습니다.")
    
    project_root = Path(__file__).resolve().parents[2]
    base = Path((config.get("models") or {}).get("root") or (project_root / "storage" / "models"))

    if os.path.isabs(local_path):
        abs_path = Path(local_path)
    else:
        s = local_path.lstrip("./")
        if "storage/models/" in s :
            suffix = s.split("storage/models/", 1)[1].strip("/")
            abs_path = (base / suffix)
        else:
            abs_path = base / os.path.basename(s)
    abs_path = abs_path.resolve()
    logger.info(f"수정된 모델경로 확인: {abs_path}")
    
    return abs_path

@register("huggingface")
def hf_factory(model_key: str) -> Streamer:
    from utils.llms.huggingface import qwen, qwen_vl, gpt_oss_20b, gemma3_27b
    from pathlib import Path

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
    if model_key.startswith("Qwen3-8B") or model_key.startswith("Qwen3-14B") :
        logger.info(f"Alibaba | Qwen Model Qwen3-8B")
        # local_path = "/home/work/CoreIQ/backend/storage/models/Qwen3-8B"
        return _Wrap(lambda messages, **kw: qwen.stream_chat(messages, model_path=local_path, **kw))

    if model_key.startswith("Qwen3-vl") :
        logger.info(f"Alibaba | Qwen Model Qwen3-vl")
        # local_path = "/home/work/CoreIQ/backend/storage/models/Qwen3-14B"
        return _Wrap(lambda messages, **kw: qwen_vl.stream_chat(messages, model_path=local_path, **kw))

    if  model_key.startswith("Qwen2.5-7B-Instruct-1M"):
        logger.info(f"Alibaba | Qwen Model Qwen2.5-7B-Instruct-1M")
        # local_path = "/home/work/CoreIQ/backend/storage/models/Qwen2.5-7B-Instruct-1M"
        return _Wrap(lambda messages, **kw: qwen.stream_chat(messages, model_path=local_path, **kw))

    if model_key.startswith("Gemma"):
        logger.info(f"Google | Gemma Model Gemma3-27B")
        return _Wrap(lambda messages, **kw: gemma3_27b.stream_chat(messages, model_path=local_path, **kw))
        
    if model_key.startswith("gpt_oss") or model_key.startswith("gpt-oss") :
        logger.info(f"gpt_oss_20b")
        # local_path = "/home/work/CoreIQ/backend/storage/models/gpt-oss-20b"
        return _Wrap(lambda messages, **kw: gpt_oss_20b.stream_chat(messages, model_path=str(local_path), **kw))
    else:
        logger.error(f"해당모델 이름으로 시작하는 로직이 없음. {model_key}")
    raise NotFoundError(f"지원하지 않는 huggingface 모델: {model_key}")


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
	def __init__(self, model: str):
		self.model = model
	def stream(self, messages, **kw):
		from utils.llms.openai.streamer import stream_chat as openai_stream
		return openai_stream(messages, model=self.model, **kw)

@register("openai")
def openai_factory(model_key: str) -> Streamer:
	return OpenAIStreamer(model_key)



