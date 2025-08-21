from typing import Callable, Dict, Generator, List, Protocol
from utils import logger

logger = logger(__name__)

# Streamer 프로토콜 정의: 메시지를 스트리밍하는 메서드를 포함
class Streamer(Protocol):
    def stream(self, messages: List[dict], **kwargs) -> Generator[str, None, None]: ...

# 프로바이더 이름을 키로 사용하여 Streamer 팩토리를 저장하는 레지스트리
_REGISTRY: Dict[str, Callable[[str], Streamer]] = {}

# 프로바이더를 레지스트리에 등록하는 데코레이터
def register(provider: str):
    def deco(factory: Callable[[str], Streamer]):
        _REGISTRY[provider.lower()] = factory
        return factory
    return deco

# 어댑터 모듈이 import되며 @register 데코레이터가 실행되도록 보장
_INITIALIZED = False
def _ensure_adapters_loaded():
    global _INITIALIZED
    if _INITIALIZED:
        return
    # 어댑터 모듈을 import하여 레지스트리에 등록
    from utils.llms import adapters
    _INITIALIZED = True

# 주어진 프로바이더와 모델 키에 따라 적절한 Streamer를 반환
def resolve(provider: str, model_key: str) -> Streamer:
    _ensure_adapters_loaded()
    try:
        return _REGISTRY[provider.lower()](model_key)
    except KeyError:
        raise ValueError(f"Unsupported provider: {provider}")

# LLM 클래스: 워크스페이스 정보를 기반으로 Streamer를 생성
class LLM:
    @staticmethod
    def from_workspace(ws: dict) -> Streamer:
        # 주어진 워크스페이스 정보에서 프로바이더와 모델 키를 사용하여 Streamer 생성
        return resolve(ws["provider"], ws["chat_model"])