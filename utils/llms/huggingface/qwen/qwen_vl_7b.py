from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

def load_qwen_vl_7b(model_dir):
    # Qwen2.5-VL 모델은 이렇게 로드
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(model_dir)
    return model, processor

if __name__ == "__main__":
    model_dir = "/home/work/CoreIQ/gpu_use/KT_sever/local_Qwen2.5-VL-7B-Instruct"
    model, processor = load_qwen_vl_7b(model_dir)
    print(model)
    print(processor)