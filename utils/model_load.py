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
