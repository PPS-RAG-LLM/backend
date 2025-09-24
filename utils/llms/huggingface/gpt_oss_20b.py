from transformers.generation.streamers import TextIteratorStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM
from threading import Thread  
from config import config
import torch, os
from utils import logger, free_torch_memory
from functools import lru_cache
from typing import List, Dict, Generator

logger = logger(__name__)

@lru_cache(maxsize=2) # 모델 로드 캐시(2개까지)
def load_gpt_oss_20b(model_dir): 
    
    # 2) 로컬 디렉터리 존재 확인
    if not os.path.isdir(model_dir):
        parent = model_dir.parent
        logger.error(f"모델 디렉터리가 없습니다: {model_dir} (parent={parent})")
        raise FileNotFoundError(f"모델 디렉터리가 없습니다: {model_dir}")
    logger.info(f"load gpt-oss-20b from `{model_dir}`")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code = True,   # 모델 코드 신뢰
        use_fast=False,             # 빠른 토크나이저 사용 여부
        local_files_only=True,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto",          # 모델 분산 처리
        torch_dtype= torch.bfloat16,      # gpt-oss-20b는 기본이 MXFP4 양자화된 MoE 경로이다.
                                    # Triton 커널이 bf16 입력을 전제해(tl.static_assert(x_format == "bf16")) 
                                    # FP16로 로드하면 컴파일/런타임이 깨짐. 
        trust_remote_code=True,     # 모델 코드 신뢰
        low_cpu_mem_usage=True,     # 메모리 효율성
        local_files_only=True,
        )
    model.eval()
    return model, tokenizer

def stream_chat(messages: List[Dict[str, str]], **gen_kwargs) -> Generator[str, None, None]:  
    logger.info(f"stream_chat: {gen_kwargs}\n\n")
    model_dir = gen_kwargs.get("model_path")
    logger.info(f"\n\nInput GPT-OSS-20b messages: {messages}\n\n")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    model, tokenizer = load_gpt_oss_20b(model_dir)
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

