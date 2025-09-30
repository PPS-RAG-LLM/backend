from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread
from functools import lru_cache          # ✅ 누락된 임포트 추가
from config import config
import time
from utils import logger, free_torch_memory
import torch

logger = logger(__name__)

@lru_cache(maxsize=2)
def load_qwen_instruct_7b(model_dir):
    logger.info(f"load qwen-7b-instruct from `{model_dir}`")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,           # Qwen 계열은 fast 토크나이저에서 템플릿 차이가 나는 경우가 있어 off 권장
    )
    # ✅ pad 토큰 보장 (없으면 EOS로 맞춤)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()
    return model, tokenizer

def stream_chat(messages, **gen_kwargs):
    logger.info(f"stream_chat: {gen_kwargs}")

    model_dir = gen_kwargs.get("model_path")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    model, tokenizer = load_qwen_instruct_7b(model_dir)

    # ✅ Qwen3 계열 권장: chat 템플릿으로 바로 인코딩
    #    add_generation_prompt=True 가 assistant 답변 시작 토큰을 자동으로 추가
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    defaults = config.get("default", {}) or {}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_args = {
        "input_ids": input_ids,
        "temperature": gen_kwargs.get("temperature", defaults.get("temperature")),
        "top_p": gen_kwargs.get("top_p", defaults.get("top_p")),
        "max_new_tokens": defaults.get("max_tokens", 512),
        "repetition_penalty": defaults.get("repetition_penalty", 1.05),
        "no_repeat_ngram_size": defaults.get("no_repeat_ngram_size", 0),
        "do_sample": True,
        "use_cache": True,
        # ✅ EOS/PAD 명시 (조기 종료/무응답 방지)
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    try:
        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()
        for text_token in streamer:
            if text_token:
                yield text_token
    finally:
        thread.join()
        free_torch_memory()


def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "user":
            prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
        elif msg["role"] == "assistant":
            prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"  # 답변 시작
    return prompt


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only Speak in Korean."},
        {"role": "user", "content": "안녕하세요. 제 이름은 김루아입니다."},
        {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
        {"role": "user", "content": "내이름이 뭐라고 했지?"},
    ]
    for chunk in stream_chat(messages):
        print(chunk, end="", flush=True)