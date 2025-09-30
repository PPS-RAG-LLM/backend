from functools import lru_cache
from utils import logger
from sentence_transformers import SentenceTransformer
from config import config as _cfg

logger = logger(__name__)

@lru_cache(maxsize=2)
def load_embedding_model():
    model_dir = (_cfg.get("user_documents", {}) or {}).get("embedding_model_dir")
    logger.info(f"load sentence model from {model_dir}")
    return SentenceTransformer(str(model_dir), device="cpu")


_MODEL_MANAGER = None

def get_model_manager():
    """지연 로딩으로 MODEL_MANAGER 가져오기"""
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from service.admin.manage_admin_LLM import _MODEL_MANAGER as MGR
        _MODEL_MANAGER = MGR
    return _MODEL_MANAGER
