import os
from openai import OpenAI
from dotenv import load_dotenv
from src.config import config

load_dotenv()

client = OpenAI(api_key=config["openai"]["api_key"])

def stream_chat(messages, **gen_kwargs):
    # gen_kwargs = {"max_tokens": 128, "temperature": 0.7, ...}
    conf = config["openai"]

    response = client.chat.completions.create(
        model       = gen_kwargs.get("model", conf["model"]),
        messages    = messages,
        stream      = True,
        max_tokens  = gen_kwargs.get("max_tokens", conf["max_tokens"]),
        temperature = gen_kwargs.get("temperature", conf["temperature"]),
        top_p       = gen_kwargs.get("top_p", conf["top_p"]),
        timeout     = gen_kwargs.get("timeout", conf["timeout"]),
    )
    for chunk in response:
        # 최신 openai 라이브러리에서는 chunk.choices[0].delta.content로 접근
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
    for chunk in stream_chat(messages):
        print(chunk, end="", flush=True)


