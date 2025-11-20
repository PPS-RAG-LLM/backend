from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from config import config as app_config
from repository.embedding_model import (
    get_active_embedding_model_name,
    get_embedding_model_path_by_name,
)
from sentence_transformers import SentenceTransformer
from utils import logger

logger = logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RETRIEVAL_PATHS = app_config.get("retrieval", {}).get("paths", {}) or {}
_USER_DOCS_CFG = app_config.get("user_documents", {}) or {}


def _resolve_model_root() -> Path:
    path_value = _RETRIEVAL_PATHS.get(
        "model_root_dir", _USER_DOCS_CFG.get("embedding_model_dir", "storage/embedding-models")
    )
    rel = Path(path_value)
    base = rel if rel.is_absolute() else (PROJECT_ROOT / rel)
    return base.resolve()


MODEL_ROOT_DIR = _resolve_model_root()

def resolve_model_input(model_key: Optional[str]) -> Tuple[str, Path]:
    """
    주어진 모델 키(embedding_models.name)를 실제 로컬 디렉토리에 매핑.
    1) DB의 model_path가 우선이며, 없을 경우 storage/embedding-models 폴더를 탐색.
    """
    key = (model_key or "bge").lower()

    try:
        db_path = get_embedding_model_path_by_name(model_key)
        if db_path:
            mp = Path(db_path).resolve()
            if mp.exists() and mp.is_dir():
                return str(model_key), mp
    except Exception:
        logger.exception("[Embedding Model] DB lookup for active model_path failed")

    candidates: List[Path] = []
    if MODEL_ROOT_DIR.exists():
        for item in MODEL_ROOT_DIR.iterdir():
            if item.is_dir():
                candidates.append(item.resolve())

    def aliases(path: Path) -> List[str]:
        name = path.name.lower()
        res = [name]
        if name.startswith("embedding_"):
            res.append(name[len("embedding_") :])
        return res

    for path in candidates:
        if key in aliases(path):
            return path.name, path
    for path in candidates:
        if key in path.name.lower():
            return path.name, path
    for path in candidates:
        if "qwen3_0_6b" in path.name.lower():
            return path.name, path

    fallback = MODEL_ROOT_DIR / "qwen3_0_6b"
    return fallback.name, fallback


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
