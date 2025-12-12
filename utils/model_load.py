import asyncio
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import config as app_config
from sentence_transformers import SentenceTransformer
from repository.rag_settings import get_rag_settings_row
from utils import logger

logger = logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RETRIEVAL_PATHS = app_config.get("retrieval", {}).get("paths", {}) or {}
_USER_DOCS_CFG = app_config.get("user_documents", {}) or {}


def _resolve_model_root() -> Path:
    path_value = app_config.get("models_dir", {}).get("embedding_model_path", "storage/models/embedding")
    rel = Path(path_value)
    base = (PROJECT_ROOT / rel)
    return base.resolve()


MODEL_ROOT_DIR = _resolve_model_root()

_EMBED_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}
_EMBED_ACTIVE_KEY: Optional[str] = None
_EMBED_LOCK = threading.Lock()

def resolve_model_input(model_key: Optional[str]) -> Tuple[str, Path]:
    """
    주어진 모델 키(embedding_models.name)를 실제 로컬 디렉토리에 매핑.
    1) DB의 model_path가 우선이며, 없을 경우 storage/models/embedding 폴더를 탐색.
    """
    key = (model_key or "bge").lower()
    try:
        embedding_key = get_rag_settings_row().get("embedding_key")
        db_path = MODEL_ROOT_DIR / embedding_key
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


def _load_hf_embedder_tuple(model_key: str) -> Tuple[Any, Any, Any]:
    from service.retrieval.common import get_or_load_hf_embedder

    _, model_dir = resolve_model_input(model_key)
    if not model_dir.exists():
        # fallback or error
        pass

    return get_or_load_hf_embedder(str(model_dir))


@lru_cache(maxsize=2)
def load_embedding_model():
    """
    DB에서 활성화된 임베딩 모델을 조회하고 SentenceTransformer 로드.
    model_path는 DB(embedding_models) 또는 폴더 스캔 fallback으로 결정됨.
    """
    model_name = get_rag_settings_row().get("embedding_key")  # 예: "embedding_qwen3_8b"
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
        from utils.llms.registry import ModelManager
        _MODEL_MANAGER = ModelManager()
    return _MODEL_MANAGER


def invalidate_embedder_cache() -> None:
    with _EMBED_LOCK:
        _EMBED_CACHE.clear()
        _EMBED_ACTIVE_KEY = None


async def get_or_load_embedder_async(model_key: str, preload: bool = False):
    """
    비동기 환경에서 임베딩 모델(토크나이저, 모델, 디바이스) 로드를 쓰레드풀로 수행.
    """
    loop = asyncio.get_running_loop()

    # 캐시가 이미 있으면 즉시 반환 (동기 체크)
    with _EMBED_LOCK:
        if _EMBED_ACTIVE_KEY == model_key and model_key in _EMBED_CACHE:
            return _EMBED_CACHE[model_key]

    return await loop.run_in_executor(None, _get_or_load_embedder, model_key, preload)


def _get_or_load_embedder(model_key: str, preload: bool = False):
    """
    전역 캐시에서 (tok, model, device) 반환.
    - 캐시에 없으면 로드해서 저장(지연 로딩)
    - preload=True는 의미상 웜업 호출일 뿐, 반환 동작은 동일
    """
    global _EMBED_CACHE, _EMBED_ACTIVE_KEY
    if not model_key:
        raise ValueError(
            "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
        )
    with _EMBED_LOCK:
        if _EMBED_ACTIVE_KEY == model_key and model_key in _EMBED_CACHE:
            return _EMBED_CACHE[model_key]
        _EMBED_CACHE.clear()
        tok, model, device = _load_hf_embedder_tuple(model_key)
        _EMBED_CACHE[model_key] = (tok, model, device)
        _EMBED_ACTIVE_KEY = model_key
        return _EMBED_CACHE[model_key]

