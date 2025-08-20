from typing import Callable, Dict, Generator, List, Protocol
from utils import logger

logger = logger(__name__)

class Streamer(Protocol):
    def stream(self, messages : List[dict], **kwargs) -> Generator[str, None, None] : ...

_REGISTRY: Dict[str, Callable[[str], Streamer]] = {}

def register(provider : str):
    def deco(factory: Callable[[str], Streamer]):
        _REGISTRY[provider.lower()] = factory
        return factory
    return deco


# 자연 초기화 : 어댑터 모듈이 import 되며 @register 데코레이터가 실행되도록 보장    
_INITIALIZED = False
def _ensure_adapters_loaded():
    global _INITIALIZED
    if _INITIALIZED:
        return
    from utils.llms.adapters import openai as _openai_adapter
    from utils.llms.adapters import huggingface as _hf_adapter
    _INITIALIZED = True

def resolve(provider : str, model_key: str) -> Streamer:
    _ensure_adapters_loaded()
    try:
        return _REGISTRY[provider.lower()](model_key)
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}")

class LLM:
    @staticmethod
    def from_workspace(ws: dict) -> Streamer:
        # logger.info(f"ws: {ws}")
        return resolve(ws["provider"], ws["chat_model"])
        