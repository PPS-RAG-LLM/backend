from functools import lru_cache
from utils import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sentence_transformers import SentenceTransformer
from config import config as _cfg
import torch, os,json
from pathlib import Path
logger = logger(__name__)

@lru_cache(maxsize=2)
def load_embedding_model():
    model_dir = (_cfg.get("user_documents", {}) or {}).get("embedding_model_dir")
    logger.info(f"load sentence model from {model_dir}")
    return SentenceTransformer(str(model_dir), device="cpu")


@lru_cache(maxsize=2) # 모델 로드 캐시(2개까지)
def load_hf_llm_model(model_dir): 
    logger.info(f"--------------------------------Before load model from {model_dir}")
    proj_base = Path(__file__).resolve().parents[1] / "storage" / "model"  # /home/work/CoreIQ/Ruah/backend/storage/model
    p = Path(str(model_dir))
    s = str(p)

    if s.startswith("/storage/model/"):
        suffix = s.split("/storage/model/", 1)[1].strip("/")
        p = proj_base / suffix                      # 여기서 config 기반 base 사용 금지(항상 proj_base)
    elif not p.is_absolute():
        # 상대경로만 config(models.root) 고려하고, 없으면 proj_base 사용
        cfg_root = (_cfg.get("models") or {}).get("root")
        base = Path(cfg_root) if cfg_root else proj_base
        p = base / s.lstrip("./")

    model_dir = str(p.resolve())
    logger.info(f"--------------------------------load model from {model_dir}")

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
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )

    model.eval()
    return model, tokenizer

# load_hf_llm_model.cache_clear()
