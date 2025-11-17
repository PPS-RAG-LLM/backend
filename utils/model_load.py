from functools import lru_cache
from pathlib import Path
from repository.embedding_model import get_active_embedding_model_name
from service.admin.manage_vator_DB import resolve_model_input
from utils import logger
from sentence_transformers import SentenceTransformer
from config import config as _cfg

logger = logger(__name__)


@lru_cache(maxsize=2)
def load_embedding_model():
    """
    DB에서 활성화된 임베딩 모델을 조회하고 SentenceTransformer 로드.
    model_path는 DB(embedding_models) 또는 폴더 스캔 fallback으로 결정됨.
    """
    model_name = get_active_embedding_model_name()  # 예: "embedding_qwen3_8b"
    _, model_dir = resolve_model_input(model_name)  # model_dir는 이미 Path 객체
    
    if not model_dir.exists():
        raise FileNotFoundError(f"임베딩 모델 경로를 찾을 수 없습니다: {model_dir}")
    
    logger.info(f"load sentence model from {model_dir}")
    return SentenceTransformer(str(model_dir), device="cuda")

_MODEL_MANAGER = None

def get_model_manager():
    """지연 로딩으로 MODEL_MANAGER 가져오기"""
    global _MODEL_MANAGER
    if _MODEL_MANAGER is None:
        from service.admin.manage_admin_LLM import _MODEL_MANAGER as MGR
        _MODEL_MANAGER = MGR
    return _MODEL_MANAGER
