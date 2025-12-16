from utils import logger
logger = logger(__name__)

def _init_client(api_key: str = None):
    from openai import OpenAI
    from dotenv import load_dotenv
    from config import config
     # 인자로 받은 api_key가 없으면 config에서 조회
    if api_key:
        final_key = api_key
    else:
        logger.debug("#### No API key in the worksapce, using config.yaml ###")
        load_dotenv()
        provider_conf = config.get("provider", {}).get("openai", {})
        final_key = provider_conf.get("api_key")
        
    if not final_key:
        # 키가 없으면 에러를 내거나 처리를 해야 함
        raise ValueError("OpenAI API Key is missing. Please check config or user settings.")

    client = OpenAI(api_key=api_key)
    return client, config.get("default", {})

def stream_chat(messages, **gen_kwargs):
    """OpenAI 채팅 스트리밍
    - model: 호출자가 지정한 모델명(예: gpt-4o-mini)
    - gen_kwargs: max_tokens, temperature, top_p, timeout 등
    """
    client, conf = _init_client()
    model_name = gen_kwargs.get("model")

    params = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": gen_kwargs.get("temperature", 0.7),
        "top_p": conf.get("top_p", 1.0),     # conf에서 안전하게 가져오기
        "timeout": conf.get("timeout", 60),  # conf에서 안전하게 가져오기
    }

     # 2. 토큰 제한값
    token_limit = conf.get("max_tokens")

    # # 3. 모델명에 따라 분기 처리 (여기가 핵심!)
    # if model_name.startswith("o1-"):
    #     # o1 모델은 max_completion_tokens 사용
    #     params["max_completion_tokens"] = token_limit
    #     # o1 모델은 보통 temperature=1 고정 권장 (선택사항)
    #     # params["temperature"] = 1.0 
    # else:
    params["max_tokens"] = token_limit

    # 4. 호출
    try:
        response = client.chat.completions.create(**params)
    except Exception as e:
        # 에러 메시지에 "max_tokens" 관련 내용이 있으면 교체해서 재시도
        err_msg = str(e).lower()
        if "max_tokens" in err_msg and "max_completion_tokens" in err_msg:
            logger.info(f"Retrying with max_completion_tokens for model {model_name}")
            params.pop("max_tokens")
            params["max_completion_tokens"] = token_limit
            response = client.chat.completions.create(**params)
        else:
            raise e

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content



if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only Speak in Korean."},
        {"role": "user", "content": "안녕하세요. 제 이름은 김루아입니다."},
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
        {"role": "user", "content": "내이름이 뭐라고 했지?"},
    ]
    for chunk in stream_chat(messages, model="gpt-4o-mini", temperature=0.7):
        print(chunk, end="", flush=True)


