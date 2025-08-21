"""Quick test script to verify local Huggingface Qwen model streaming.

Run:
    python backend/service/admin/test.py

Prerequisites:
1. `backend/storage/model/qwen-7b` (또는 qwen-2.5-7b-instruct) 폴더에 모델 파일 존재
2. `llm_models` 테이블에 provider="huggingface", name="qwen-7b" (또는 아래 chat_model)에 해당하는 row 존재
3. 필수 패키지: transformers, accelerate, peft, etc.
"""

import sys
from utils.llms.registry import LLM


def main() -> None:
    # Workspace dummy – matches llm_models row
    ws = {
        "provider": "huggingface",
        "chat_model": "qwen-7b",  # change if your DB row uses different key
    }

    streamer = LLM.from_workspace(ws)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only speak Korean."},
        {"role": "user", "content": "한국에서 가장 높은 산은 어디인가요?"},
    ]

    print("Streaming response:\n")
    try:
        for tok in streamer.stream(messages, temperature=0.7):
            sys.stdout.write(tok)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
