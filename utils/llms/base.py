# utils/llm/base.py
from typing import List, Dict, Any, Generator, Callable
from config import config
from utils.llms.openai.streamer import stream_chat as openai_stream
from utils.llms.huggingface.qwen import qwen_7b, qwen_vl_7b

def _resolve_model(provider: str, model_key: str) -> Dict[str, Any]:
    provider = provider.lower()
    aliases = config.get("aliases", {})
    real_key = aliases.get(model_key, model_key)
    reg = config.get("registry", {})
    if provider not in reg:
        raise ValueError(f"레지스트리에 provider가 없습니다: {provider}")
    if real_key not in reg[provider]:
        raise ValueError(f"레지스트리에 모델 키가 없습니다: {provider}/{model_key}")
    return reg[provider][real_key]

def _merge(base: Dict[str, Any] | None, override: Dict[str, Any] | None) -> Dict[str, Any]:
    out = dict(base or {})
    out.update(override or {})
    return out

class CallableModel:
    def __init__(self, fn: Callable[..., Generator[str, None, None]]):
        self._fn = fn
    def stream_chat(self, messages: List[Dict], **overrides) -> Generator[str, None, None]:
        return self._fn(messages, **overrides)

def model_factory(provider: str, model_key: str) -> CallableModel:
    info = _resolve_model(provider, model_key)
    provider = provider.lower()
    params = info.get("params", {})

    if provider == "openai":
        api_model = info.get("api_model")
        def run(messages, **overrides):
            return openai_stream(messages, model=api_model, **_merge(params, overrides))
        return CallableModel(run)

    if provider == "huggingface":
        family = info.get("family")
        local_path = info.get("local_path")
        if family == "qwen":
            def run(messages, **overrides):
                return qwen_7b.stream_chat(messages, model_path=local_path, **_merge(params, overrides))
            return CallableModel(run)
        if family == "qwen-vl":
            def run(messages, **overrides):
                return qwen_vl_7b.stream_chat(messages, model_path=local_path, **_merge(params, overrides))
            return CallableModel(run)
        raise ValueError(f"지원하지 않는 huggingface family: {family}")

    if provider == "ollama":
        # TODO: utils.llm.ollama.streamer 연결
        raise NotImplementedError("ollama 프로바이더는 아직 지원되지 않습니다.")

    raise ValueError(f"지원하지 않는 모델 제공자: {provider}")