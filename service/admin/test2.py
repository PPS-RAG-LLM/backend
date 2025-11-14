"""Test script to stream response from an OSS/Huggingface model (e.g., GPT-OSS-20B).

Usage:
python service/admin/test2.py

If arguments are omitted, defaults are used.

Requirements:
1. The specified model must exist in `llm_models` table with provider='huggingface'.
2. Model files must be placed under `backend/storage/models/<model_name>/`.
3. Necessary packages: transformers, accelerate, peft, bitsandbytes, etc.
"""

import sys
from utils.llms.registry import LLM


def main():
    # Parse CLI args
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gpt_oss_20b"
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Open-source GPT 모델의 파라미터 수는 얼마인가요?"

    ws = {
        "provider": "huggingface",  # local OSS models are handled via Huggingface pipeline
        "chat_model": model_name,
    }

    streamer = LLM.from_workspace(ws)

    messages = [
        {"role": "system", "content": "You are a knowledgeable assistant. Answer in Korean."},
        {"role": "user", "content": question},
    ]

    print(f"\n[Model: {model_name}] 질문: {question}\n응답 (stream):\n")
    try:
        for tok in streamer.stream(messages, temperature=0.7):
            sys.stdout.write(tok)
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[interrupted]")


if __name__ == "__main__":
    main()
