from transformers.generation.streamers import TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread, RLock
from config import config
import torch, os
from utils import logger, free_torch_memory
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Generator
from utils.model_load import get_model_manager

logger = logger(__name__)
_load_guard = RLock()  # 동일 경로 동시 로딩 방지

def _norm_path(p) -> str:
    """항상 동일한 캐시 키(절대경로 문자열)로 정규화."""
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(p)


@lru_cache(maxsize=2)
def _load_impl(norm_dir: str):
    # 여긴 '한 번만' 실제 로드되는 자리
    if not os.path.isdir(norm_dir):
        raise FileNotFoundError(f"모델 디렉터리가 없습니다: {norm_dir}")
    logger.info(f"load gpt-oss-20b from `{norm_dir}`")
    with _load_guard:
        tok = AutoTokenizer.from_pretrained(
            norm_dir, trust_remote_code=True, use_fast=False, local_files_only=True
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        mdl = AutoModelForCausalLM.from_pretrained(
            norm_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        mdl.eval()
        return mdl, tok

def load_gpt_oss_20b(model_dir):
    return _load_impl(_norm_path(model_dir))

def stream_chat(messages: List[Dict[str, str]], **gen_kwargs) -> Generator[str, None, None]:  
    logger.info(f"stream_chat: {gen_kwargs}\n\n")
    model_dir = gen_kwargs.get("model_path")
    logger.info(f"\n\nInput GPT-OSS-20b messages: {messages}\n\n")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    model, tokenizer = load_gpt_oss_20b(model_dir)  # ← 리턴 순서 주의: (model, tokenizer)

    # 1. Harmony chat template 자동 적용
    #    - add_generation_prompt=True: assistant 응답 시작에 맞춰 템플릿 완성
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize= True,
        add_generation_prompt=True,
        return_tensors= "pt"
    ).to(model.device)


    # 2. 공식 권장 샘플링값 반영
    defaults        = config.get("default") 
    streamer        = TextIteratorStreamer(        # github: gpt_oss 스트리머 설정
        tokenizer, 
        skip_prompt=True, 
        skip_special_tokens=True, 
    )
    generation_args = {
        "input_ids": input_ids,
        "temperature": gen_kwargs.get("temperature"),
        "max_new_tokens": defaults.get("max_tokens"),
        "top_p": gen_kwargs.get("top_p"),
        "do_sample": True,
        "repetition_penalty": defaults.get("repetition_penalty"),
        "no_repeat_ngram_size": defaults.get("no_repeat_ngram_size"),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "streamer": streamer,
    }
    try:
        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.start()
        for text_token in streamer:
            if text_token:
                yield text_token
    except Exception as e:
        logger.exception({"event": "streaming_runtime_error", "error": str(e)})
        raise
    finally:
        thread.join()
        free_torch_memory()  # 캐시만 비움(모델 유지)


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