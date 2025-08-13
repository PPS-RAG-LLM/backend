from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer
from transformers.generation.configuration_utils import GenerationConfig
from threading import Thread  
from config import config
import time 

def load_gpt_oss_20b(model_dir): 
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return model, tokenizer


def stream_chat(messages, model_path, **gen_kwargs):  

    model, tokenizer = load_gpt_oss_20b(model_path)
    text = build_qwen_prompt(messages)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model_inputs = tokenizer(text, return_tensors="pt", )
    for k, v in model_inputs.items():
        model_inputs[k] = v.to(model.device)
    
    conf = config["registry"]["gpt-oss-20b"]
        
    generation_args = {
        "max_new_tokens": gen_kwargs.get("max_new_tokens", conf["max_new_tokens"]),
        "temperature": gen_kwargs.get("temperature", conf["temperature"]),
        "top_p": gen_kwargs.get("top_p", conf["top_p"]),
        "repetition_penalty": gen_kwargs.get("repetition_penalty", conf["repetition_penalty"]),
        "no_repeat_ngram_size": gen_kwargs.get("no_repeat_ngram_size", conf["no_repeat_ngram_size"]),
        "early_stopping": gen_kwargs.get("early_stopping", conf["early_stopping"]),
        "streamer": streamer,
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

