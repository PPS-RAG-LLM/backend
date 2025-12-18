from utils import logger

logger = logger(__name__)

def _init_client(api_key: str = None):
    import anthropic
    from config import config

    # API 키 확인
    if not api_key:
        raise ValueError("Anthropic API Key is missing. Please check user settings.")

    client = anthropic.Anthropic(api_key=api_key)
    return client, config.get("default", {})

def _extract_system_message(messages):
    """messages 리스트에서 system 메시지를 분리하여 별도 문자열로 반환"""
    system_prompts = []
    filtered_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_prompts.append(msg["content"])
        else:
            filtered_messages.append(msg)
            
    # 여러 개의 시스템 메시지가 있다면 하나로 합침
    system_instruction = "\n".join(system_prompts) if system_prompts else None
    return system_instruction, filtered_messages

def stream_chat(messages, **gen_kwargs):
    """Anthropic Claude 채팅 스트리밍"""
    api_key = gen_kwargs.get("api_key")
    client, conf = _init_client(api_key)
    
    model_name = gen_kwargs.get("model", "claude-3-5-sonnet-20240620")

    # 1. System 메시지 분리 (Claude API 요구사항)
    system_instruction, filtered_messages = _extract_system_message(messages)

    # 2. 파라미터 구성
    # max_tokens가 없으면 기본값 설정 (Claude는 필수값인 경우가 많음)
    max_tokens = conf.get("max_tokens", 4096)
    if "max_tokens" in gen_kwargs:
        max_tokens = gen_kwargs["max_tokens"]

    params = {
        "model": model_name,
        "messages": filtered_messages,
        "max_tokens": max_tokens,
        "temperature": gen_kwargs.get("temperature", 0.7),
    }
    
    if system_instruction:
        params["system"] = system_instruction

    # 3. API 호출 및 스트리밍
    try:
        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
                
    except Exception as e:
        logger.error(f"Anthropic API Error: {e}")
        raise e

if __name__ == "__main__":
    # 테스트 코드
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Speak Korean."},
        {"role": "user", "content": "안녕, 클로드!"},
    ]
    # import os
    # api_key = os.getenv("ANTHROPIC_API_KEY")
    # if api_key:
    #     for chunk in stream_chat(messages, api_key=api_key, model="claude-3-5-sonnet-20240620"):
    #         print(chunk, end="", flush=True)