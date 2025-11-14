from functools import lru_cache
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.generation.streamers import TextIteratorStreamer
from qwen_vl_utils import process_vision_info
import torch
from utils import logger, free_torch_memory
from threading import Thread
from config import config

logger = logger(__name__)

# PyTorch Dynamo 비활성화 (호환성 이슈 우회)
torch._dynamo.config.disable = True


@lru_cache(maxsize=2)
def load_qwen3_vl_32b(model_dir):
    """
    Qwen3-VL-32B 멀티모달 모델 로딩 (이미지 + 비디오 + 텍스트)
    참고: https://github.com/QwenLM/Qwen3-VL
    """
    logger.info(f"load Qwen3-VL-32B from `{model_dir}`")

    # Processor 로딩 (이미지 + 비디오 + 텍스트 통합 전처리)
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )

    # Vision-Language 모델 로딩 (AutoModelForVision2Seq가 자동으로 Qwen3VL 클래스 선택)
    model = AutoModelForVision2Seq.from_pretrained(
        model_dir,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    model.eval()
    return model, processor


def stream_chat(messages, **gen_kwargs):
    """
    Qwen3-VL 스트리밍 채팅 (멀티모달: 이미지 + 비디오 + 텍스트)
    
    Args:
        messages: OpenAI 포맷 메시지 리스트
            예시: [{"role": "user", "content": [
                {"type": "image", "image": "url_or_path"},
                {"type": "text", "text": "이미지에 대해 설명해주세요"}
            ]}]
        **gen_kwargs: 생성 파라미터 (temperature, top_p, model_path 등)
    
    Yields:
        str: 스트리밍 채팅 조각
    """
    logger.info(f"stream_chat: {gen_kwargs}")

    model_dir = gen_kwargs.get("model_path")
    if not model_dir:
        raise ValueError("누락된 파라미터: config.yaml의 model_path")

    model, processor = load_qwen3_vl_32b(model_dir)
    
    defaults = config.get("default", {}) or {}
    
    # 1️⃣ 텍스트 프롬프트 생성
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 2️⃣ 이미지/비디오 전처리 (qwen_vl_utils 사용)
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 3️⃣ 최종 입력 준비
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 4️⃣ 스트리머 생성
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_args = {
        **inputs,
        "temperature": gen_kwargs.get("temperature", defaults.get("temperature", 0.7)),
        "top_p": gen_kwargs.get("top_p", defaults.get("top_p", 0.8)),
        "top_k": defaults.get("top_k", 20),
        "max_new_tokens": defaults.get("max_tokens", 1024),
        "repetition_penalty": defaults.get("repetition_penalty", 1.0),
        "do_sample": True,
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


#####  모델 다운로드 스크립트 (반드시 해당 폴더 내에서 실행) #####
# python - <<'EOF'
# from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq
# import torch

# model_id = "Qwen/Qwen3-VL-32B-Instruct"
# print(f"🔽 Downloading multimodal model: {model_id}")

# # 1️⃣ Processor: (이미지 + 텍스트 통합 전처리기)
# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# # 2️⃣ Vision-Language 모델 로드
# model = AutoModelForVision2Seq.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto",
# )

# # 3️⃣ 저장 (현재 디렉토리)
# processor.save_pretrained("./")
# model.save_pretrained("./")

# print("✅ Done. Qwen3-VL-32B-Instruct 다운로드 완료 (현재 폴더에 저장됨)")
# EOF