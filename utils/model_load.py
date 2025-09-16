from functools import lru_cache
from utils import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from config import config as _cfg
import torch

logger = logger(__name__)

@lru_cache(maxsize=2)
def load_embedding_model():
    model_dir = (_cfg.get("user_documents", {}) or {}).get("embedding_model_dir")
    logger.info(f"load sentence model from {model_dir}")
    return SentenceTransformer(str(model_dir), device="cpu")


@lru_cache(maxsize=2) # 모델 로드 캐시(2개까지)
def load_hf_llm_model(model_dir): 
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code = True,   # 모델 코드 신뢰
        use_fast=False,             # 빠른 토크나이저 사용 여부
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto",          # 모델 분산 처리
        torch_dtype= torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,     # 모델 코드 신뢰
        low_cpu_mem_usage=True      # 메모리 효율성
        )
    model.eval()
    return model, tokenizer