from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread  
from config import config
import time, torch
from importlib import import_module
from utils import logger
from functools import lru_cache
from typing import List, Dict, Any, Generator

logger = logger(__name__)

@lru_cache(maxsize=2) # 모델 로드 캐시(2개까지)
def load_gpt_oss_20b(model_dir): 
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
            use_fast=False,
            padding_side="left",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # ----------------------------
        # 1) 새 transformers(>=4.46) : Mxfp4Config 사용
        # 2) 구버전               : legacy load_in_4bit kwargs 사용
        # ----------------------------
        try:
            Mxfp4Config = getattr(
                import_module("transformers.utils.quantization_config"),
                "Mxfp4Config",
            )
            quant_cfg = Mxfp4Config(
                compute_dtype=torch.bfloat16,
                weight_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=quant_cfg,
            )
        except (AttributeError, ModuleNotFoundError):
            # fallback for older transformers
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        try:
            model.to(torch.bfloat16)
        except Exception:
            # some older accelerate versions disallow .to after device_map
            pass
        model.eval()
        return model, tokenizer
    except Exception as e:
        logger.exception({"event": "load_gpt_oss_20b_failed", "error": str(e)})
        raise


def stream_chat(messages: List[Dict[str, str]], **gen_kwargs) -> Generator[str, None, None]:  
    model_dir = gen_kwargs.get("model_path")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    try:
        model, tokenizer = load_gpt_oss_20b(model_dir)
    except Exception as e:
        logger.error({"event": "stream_chat_failed", "stage": "load", "error": str(e)})
        raise

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
    except Exception as e:
        logger.exception({"event": "generation_start_failed", "error": str(e)})
        raise

    #3. 스트리망 토큰 yield
    try:
        for text_token in streamer:
            if text_token:
                yield text_token
    except Exception as e:
        logger.exception({"event": "streaming_runtime_error", "error": str(e)})
        raise
    finally:
        thread.join()

def build_qwen_prompt(messages):
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

