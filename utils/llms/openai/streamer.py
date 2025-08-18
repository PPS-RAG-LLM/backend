import os
from openai import OpenAI
from dotenv import load_dotenv
from config import config

load_dotenv()

client = OpenAI(api_key=config["provider"]["openai"]["api_key"])

def stream_chat(messages, **gen_kwargs):
    """OpenAI 채팅 스트리밍
    - model: 호출자가 지정한 모델명(예: gpt-4o-mini)
    - gen_kwargs: max_tokens, temperature, top_p, timeout 등
    """
    conf = config["default"]
    response = client.chat.completions.create(
        model       = gen_kwargs.get("model"),
        messages    = messages,
        stream      = True,
        temperature = gen_kwargs.get("temperature"),
        max_tokens  = conf["max_tokens"],
        top_p       = conf["top_p"],
        timeout     = conf["timeout"],
    )
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


