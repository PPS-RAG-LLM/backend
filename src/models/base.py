from abc    import ABC, abstractmethod
from typing import Generator, List, Dict
# 각 모델의 기존 스트리머 가져오기
from src.models.openai.streamer import stream_chat as openai_stream
from src.models.qwen.streamer   import stream_chat as qwen_stream
from src.config import config

class BaseModel(ABC):
    @abstractmethod
    def stream_chat(self, messages: List[Dict]) -> Generator[str, None, None]:
        """LLM 모델과의 스트리밍 대화 생성"""
        ...

# ----- 모델별 어댑터 -------------------------------------------------
class OpenAIModel(BaseModel):
    def stream_chat(self, messages: List[Dict]) -> Generator[str, None, None]:
        return openai_stream(messages)


class QwenModel(BaseModel):
    def stream_chat(self, messages: List[Dict]) -> Generator[str, None, None]:
        return qwen_stream(messages)
# -------------------------------------------------------------------


def model_factory(name: str) -> BaseModel:
    """모델 문자열 → 어댑터 인스턴스 반환"""
    name = name.lower()

    # config에서 모델 목록 가져오기
    openai_models = [model.lower() for model in config.get("models", {}).get("openai", [])]
    qwen_models = [model.lower() for model in config.get("models", {}).get("qwen", [])]

    if name in openai_models:
        return OpenAIModel()
    if name in qwen_models:
        return QwenModel()

    # 지원되는 모델 목록 표시
    all_models = openai_models + qwen_models
    raise ValueError(f"지원하지 않는 모델: {name}")