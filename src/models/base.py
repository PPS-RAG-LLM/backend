from abc    import ABC, abstractmethod
from typing import Generator, List, Dict
# 각 모델의 기존 스트리머 가져오기
from src.models.openai.streamer import stream_chat as openai_stream
from src.models.qwen.streamer   import stream_chat as qwen_stream


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

openai_model = ["gpt-4o-mini", "gpt-4o"]
qwen_model = ["qwen-2.5-7b-instruct"]

def model_factory(name: str) -> BaseModel:
    """모델 문자열 → 어댑터 인스턴스 반환"""
    name = name.lower()
    if name in openai_model:
        return OpenAIModel()
    if name in qwen_model:
        return QwenModel()
    raise ValueError(f"지원하지 않는 모델: {name}")