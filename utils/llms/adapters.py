
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


### Local (fine-tuned) provider
# 파인튜닝된 모델은 provider="local"로 DB에 저장됨.
# 모델 이름 패턴에 따라 적절한 HuggingFace 스트리머를 자동 선택한다.


def _detect_model_family(model_key: str, model_dir: str) -> str:
    """
    모델 이름 및 디렉토리의 config 파일을 분석하여 모델 패밀리를 감지한다.
    반환값: "qwen", "gemma", "gpt_oss", "unknown"
    """
    import json

    lower_key = model_key.lower()

    # 1단계: 모델 이름 패턴 매칭
    if "qwen" in lower_key:
        return "qwen"
    if "gemma" in lower_key:
        return "gemma"
    if "gpt_oss" in lower_key or "gpt-oss" in lower_key:
        return "gpt_oss"

    # 2단계: config.json / tokenizer_config.json 파일에서 모델 아키텍처 감지
    config_candidates = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "tokenizer_config.json"),
    ]
    combined_text = ""
    for cfg_path in config_candidates:
        try:
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                combined_text += " " + json.dumps(data).lower()
        except Exception as e:
            logger.warning("모델 config 파일 읽기 실패 (%s): %s", cfg_path, e)

    if combined_text:
        if "qwen" in combined_text:
            logger.info("config 파일에서 Qwen 계열 감지: %s", model_key)
            return "qwen"
        if "gemma" in combined_text:
            logger.info("config 파일에서 Gemma 계열 감지: %s", model_key)
            return "gemma"
        if "gpt_oss" in combined_text or "gpt-oss" in combined_text:
            logger.info("config 파일에서 GPT-OSS 계열 감지: %s", model_key)
            return "gpt_oss"

    logger.warning("모델 패밀리 감지 불가: %s (디렉토리: %s)", model_key, model_dir)
    return "unknown"


@register("local")
def local_factory(model_key: str) -> Streamer:
    from utils.llms.huggingface import qwen, gpt_oss_20b, gemma3_27b
    from pathlib import Path

    if model_key in _HF_STREAMER_CACHE:
        logger.debug("local_factory cache hit: %s", model_key)
        return _HF_STREAMER_CACHE[model_key]

    logger.info(f"local_factory: {model_key}")

    # DB에서 provider="local"인 모델 정보 조회
    model_info = get_llm_model_by_provider_and_name("local", model_key)
    if not model_info:
        # 이름만으로 폴백 조회 시도
        from repository.llm_models import repo_get_llm_model_by_name
        model_info = repo_get_llm_model_by_name(model_key)
        if not model_info:
            raise NotFoundError(f"로컬 모델을 찾을 수 없습니다: {model_key}")

    logger.info(f"local model_info: {model_info}")

    local_path = _resolve_model_path(model_info.get("model_path"))
    if not os.path.isdir(local_path):
        raise NotFoundError(f"모델 디렉토리를 찾을 수 없습니다: {local_path}")

    # 모델 이름 또는 config 파일 기반으로 모델 패밀리 감지 후 적절한 Streamer 생성
    streamer: Optional[Streamer] = None
    family = _detect_model_family(model_key, str(local_path))

    if family == "qwen":
        logger.info("Local | Qwen 계열 모델: %s", model_key)
        streamer = _Wrap(lambda messages, **kw: qwen.stream_chat(messages, model_path=local_path, **kw))

    elif family == "gemma":
        logger.info("Local | Gemma 계열 모델: %s", model_key)
        streamer = _Wrap(lambda messages, **kw: gemma3_27b.stream_chat(messages, model_path=local_path, **kw))

    elif family == "gpt_oss":
        logger.info("Local | GPT-OSS 모델: %s", model_key)
        streamer = _Wrap(lambda messages, **kw: gpt_oss_20b.stream_chat(messages, model_path=str(local_path), **kw))

    else:
        # 감지 불가 시 Qwen 로더를 기본 폴백으로 사용
        logger.warning("Local 모델 패밀리 감지 불가, Qwen 로더로 폴백: %s (감지결과: %s)", model_key, family)
        streamer = _Wrap(lambda messages, **kw: qwen.stream_chat(messages, model_path=local_path, **kw))

    _HF_STREAMER_CACHE[model_key] = streamer
    return streamer


### Base Streamer

class BaseAPIStreamer:
    def __init__(self, default_model: str, api_key: str = None):
        self.default_model = default_model
        self.api_key = api_key

    def _get_stream_function(self):
        """자식 클래스에서 구현할 추상 메서드"""
        raise NotImplementedError

    def stream(self, messages, **kw):
        stream_func = self._get_stream_function()
        
        if "model" not in kw or not kw["model"]:
            kw["model"] = self.default_model
        
        # 생성자에서 받은 api_key 주입 (우선순위: kw > 생성자)
        if self.api_key and "api_key" not in kw:
            kw["api_key"] = self.api_key
            
        return stream_func(messages, **kw)


### OpenAI API

class OpenAIStreamer(BaseAPIStreamer):
    
    def _get_stream_function(self):
        from utils.llms.openai.streamer import stream_chat as openai_stream
        return openai_stream
@register("openai")
def openai_factory(model_key: str, **kwargs) -> Streamer:
    api_key = kwargs.get("api_key")
    return OpenAIStreamer(default_model=model_key, api_key=api_key)


### Gemini API

class GeminiStreamer(BaseAPIStreamer):
    def  _get_stream_function(self):
        from utils.llms.gemini.streamer import stream_chat as gemini_stream
        return gemini_stream

@register("gemini")
@register("google")
def gemini_factory(model_key: str, **kwargs) -> Streamer:
    api_key = kwargs.get("api_key")
    return GeminiStreamer(default_model=model_key, api_key=api_key)


### Anthropic (Claude) API

class AnthropicStreamer(BaseAPIStreamer):
    def _get_stream_function(self):
        from utils.llms.anthropic.streamer import stream_chat as anthropic_stream
        return anthropic_stream

@register("anthropic")
@register("claude")
def anthropic_factory(model_key: str, **kwargs) -> Streamer:
    api_key = kwargs.get("api_key")
    return AnthropicStreamer(default_model=model_key, api_key=api_key)