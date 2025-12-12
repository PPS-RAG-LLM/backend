from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
# from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread
from functools import lru_cache          # ✅ 누락된 임포트 추가
from config import config
# import time
from utils import logger, free_torch_memory
import torch

logger = logger(__name__)

@lru_cache(maxsize=2)
def load_qwen_instruct_7b(model_dir):
    logger.info(f"load qwen-instruct-model from `{model_dir}`")
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

    # 🎯 파인튜닝된 모델 호환성 개선 (FULL/LORA/QLORA 지원)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 파인튜닝과 동일한 dtype 사용
        low_cpu_mem_usage=True,      # 메모리 최적화
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
    )
    
    # 🎯 Multi-GPU 모델 처리 개선
    try:
        # Multi-GPU 모델의 경우 첫 번째 device 사용
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_device = next(iter(model.hf_device_map.values()))
            input_ids = input_ids.to(first_device)
        else:
            input_ids = input_ids.to(model.device)
    except Exception:
        # Fallback: GPU 0 사용
        input_ids = input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")

    defaults = config.get("default", {}) or {}
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 🎯 파인튜닝된 모델 호환 generation parameters
    generation_args = {
        "input_ids": input_ids,
        "max_new_tokens": gen_kwargs.get("max_new_tokens", defaults.get("max_tokens", 512)),
        "do_sample": True,
        "temperature": gen_kwargs.get("temperature", defaults.get("temperature", 0.7)),
        "top_p": gen_kwargs.get("top_p", defaults.get("top_p", 0.8)),
        "repetition_penalty": gen_kwargs.get("repetition_penalty", defaults.get("repetition_penalty", 1.05)),
        "streamer": streamer,
        # ✅ 파인튜닝된 모델 안정성 개선
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    
    # 🛡️ 파인튜닝된 모델에서 문제되는 파라미터 제거
    no_repeat_ngram = defaults.get("no_repeat_ngram_size", 0)
    if no_repeat_ngram and no_repeat_ngram > 0:
        generation_args["no_repeat_ngram_size"] = no_repeat_ngram

    # 🎯 파인튜닝된 Multi-GPU 모델 안정적 처리
    try:
        logger.debug(f"Starting generation with device: {input_ids.device}")
        thread = Thread(target=model.generate, kwargs=generation_args)
        thread.daemon = True  # 메인 프로세스 종료 시 같이 종료
        thread.start()
        
        # 🛡️ 무한 대기 방지를 위한 타임아웃 처리
        import time
        start_time = time.time()
        timeout_seconds = 300  # 5분 타임아웃
        
        for text_token in streamer:
            if text_token:
                yield text_token
            
            # 타임아웃 체크
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Generation timeout after {timeout_seconds}s - terminating")
                break
                
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        yield f"[오류] 생성 중 문제가 발생했습니다: {str(e)}"
    finally:
        try:
            # 스레드 정리 (타임아웃과 함께)
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("Generation thread did not terminate cleanly")
        except Exception:
            pass
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