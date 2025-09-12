from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread  
from config import config
import time 
from utils import logger, free_torch_memory, load_hf_llm_model
from functools import lru_cache
import torch

logger = logger(__name__)

# @lru_cache(maxsize=2) # 모델 로드 캐시(2개까지)
# def load_qwen_instruct_7b(model_dir): 
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_dir, 
#         local_files_only=True, 
#         trust_remote_code=True
#         )
#     model = AutoModelForCausalLM.from_pretrained(
#         model_dir, 
#         device_map="auto", 
#         local_files_only=True, 
#         trust_remote_code=True,
#         torch_dtype= torch.float16 if torch.cuda.is_available() else torch.float32,
#         )
#     model.eval()
#     return model, tokenizer

def stream_chat(messages, **gen_kwargs):  
    logger.info(f"stream_chat: {gen_kwargs}")

    model_dir = gen_kwargs.get("model_path")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    model, tokenizer = load_hf_llm_model(model_dir)
    text = build_qwen_prompt(messages)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model_inputs = tokenizer(text, return_tensors="pt", )
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)
        
    defaults = config.get("default", {}) or {}

    generation_args = {
        "temperature"           : gen_kwargs.get("temperature"),
        "max_new_tokens"        : defaults.get("max_tokens"),
        "top_p"                 : defaults.get("top_p"),
        "repetition_penalty"    : defaults.get("repetition_penalty"),
        "no_repeat_ngram_size"  : defaults.get("no_repeat_ngram_size"),
        "early_stopping"        : defaults.get("early_stopping"),
        "streamer"              : streamer,
        **model_inputs
    }
    thread = Thread(
        target=model.generate,
        kwargs=generation_args,
    )
    thread.start()
    acc_text = ""               # Accumulate and yield text tokens as they are generated
    for text_token in streamer:
        time.sleep(0.0001)      # Simulate real-time output with a short delay
        yield  text_token       # Yield the accumulated text
    thread.join()
    free_torch_memory()

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