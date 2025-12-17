from google import genai
from google.genai import types
from utils import logger
from config import config

logger = logger(__name__)

def _init_client(api_key: str = None):
    # [수정] config에서 조회하는 로직 제거. api_key가 없으면 즉시 에러 발생.
    if not api_key:
        raise ValueError("Gemini API Key is missing. Please check user settings.")

    client = genai.Client(api_key=api_key)
    return client, config.get("default", {})

def _convert_messages(messages):
    """OpenAI 스타일의 messages를 Gemini contents 포맷으로 변환"""
    gemini_contents = []
    system_instruction = None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "system":
            system_instruction = content
        elif role == "user":
            gemini_contents.append(types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            gemini_contents.append(types.Content(role="model", parts=[types.Part(text=content)]))
            
    return gemini_contents, system_instruction

def stream_chat(messages, **gen_kwargs):
    """Gemini 채팅 스트리밍
    - model: 호출자가 지정한 모델명 (예: gemini-3-pro-preview)
    - gen_kwargs: max_tokens, temperature, top_p 등
    """
    api_key = gen_kwargs.get("api_key")
    
    client, conf = _init_client(api_key)
    model_name = gen_kwargs.get("model", "gemini-3-pro-preview")

    contents, system_instruction = _convert_messages(messages)

    # 파라미터 매핑
    token_limit = conf.get("max_tokens")
    if "max_tokens" in gen_kwargs:
        token_limit = gen_kwargs["max_tokens"]

    config_params = {
        "temperature": gen_kwargs.get("temperature", 0.7),
        "top_p": conf.get("top_p", 0.95),
        "max_output_tokens": token_limit,
    }
    
    if system_instruction:
        config_params["system_instruction"] = system_instruction

    generate_config = types.GenerateContentConfig(**config_params)

    try:
        response = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_config
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        raise e

if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only Speak in Korean."},
        {"role": "user", "content": "안녕하세요. 제 이름은 김루아입니다."},
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
        {"role": "user", "content": "내이름이 뭐라고 했지?"},
    ]
    # 테스트 시에는 실제 API 키가 필요합니다.
    # import os
    # api_key = os.getenv("GEMINI_API_KEY")
    # if api_key:
    #     for chunk in stream_chat(messages, api_key=api_key, model="gemini-3-pro-preview"):
    #         print(chunk, end="", flush=True)

