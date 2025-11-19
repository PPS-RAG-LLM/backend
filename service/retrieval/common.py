"""공통 검색/임베딩 유틸리티."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoModel, AutoTokenizer  # type: ignore

from utils import free_torch_memory, load_embedding_model, logger

logger = logger(__name__)

# -----------------------------------------------------------------------------
# 경로/캐시 헬퍼
# -----------------------------------------------------------------------------
_DOC_INFO_DIR: Optional[Path] = None
_VECTOR_CACHE_DIR: Optional[Path] = None

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}
_MODEL_CACHE: Dict[str, AutoModel] = {}
_CACHE_LOCK = threading.Lock()


def _doc_dirs() -> Tuple[Path, Path]:
    """config.yaml 기반 문서 메타/벡터 디렉토리 경로 반환."""
    global _DOC_INFO_DIR, _VECTOR_CACHE_DIR
    if _DOC_INFO_DIR is not None and _VECTOR_CACHE_DIR is not None:
        return _DOC_INFO_DIR, _VECTOR_CACHE_DIR

    from config import config as _cfg  # 지연 임포트

    base = _cfg.get("user_documents", {}) or {}
    _DOC_INFO_DIR = Path(base.get("doc_info_dir", "storage/documents/documents-info"))
    _VECTOR_CACHE_DIR = Path(base.get("vector_cache_dir", "storage/documents/vector-cache"))
    return _DOC_INFO_DIR, _VECTOR_CACHE_DIR


def get_document_dirs() -> Tuple[Path, Path]:
    """외부 모듈에서도 사용할 수 있는 경로 헬퍼."""
    return _doc_dirs()


# -----------------------------------------------------------------------------
# 로컬 문서 메타/벡터 로딩
# -----------------------------------------------------------------------------
def get_document_title(doc_id: str) -> str:
    """doc_id에 해당하는 문서 제목 반환 (없으면 fallback)."""
    doc_info_dir, _ = _doc_dirs()
    for path in doc_info_dir.glob(f"*{doc_id}.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - 로깅만
            logger.error("Failed to load doc info (%s): %s", path, exc)
            continue
        title = data.get("title")
        if title:
            return str(title)
        # title이 없으면 파일명에서 UUID를 제거하고 반환
        return path.stem.rsplit("-", 5)[0]
    return "Unknown Document"


def load_document_vectors(doc_id: str) -> List[Dict[str, Any]]:
    """doc_id(json) 벡터 캐시 로드. 파일명이 다를 경우 doc_id 포함 파일을 탐색."""
    _, vec_dir = _doc_dirs()
    primary_path = vec_dir / f"{doc_id}.json"
    path = primary_path

    if not primary_path.exists():
        # 제목-UUID 형태 파일명을 고려해 doc_id를 포함하는 파일 검색
        matches = sorted(vec_dir.glob(f"*{doc_id}.json"))
        if matches:
            path = matches[0]
            logger.debug(
                "[VectorLoad] doc_id=%s matched file=%s", doc_id, path.name
            )
        else:
            logger.warning(
                "[VectorLoad] no vector cache found for doc_id=%s (searched %s)",
                doc_id,
                primary_path,
            )
            return []

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load vectors for %s (path=%s): %s", doc_id, path, exc)
        return []


# -----------------------------------------------------------------------------
# 임베딩/유사도
# -----------------------------------------------------------------------------
def get_embedding_model():
    """
    SentenceTransformer 스타일 임베딩 모델 객체 반환.

    utils.load_embedding_model()의 결과를 그대로 노출하되 내부 캐싱을 활용한다.
    """
    return load_embedding_model()


def embed_text(text: str) -> np.ndarray:
    """
    단일 문장을 임베딩하여 numpy 벡터로 반환.
    SentenceTransformer 호환 모델 기준.
    """
    model = get_embedding_model()
    result = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    free_torch_memory()
    return result


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도 계산."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float((a @ b) / (na * nb))


# -----------------------------------------------------------------------------
# HuggingFace 기반 임베더 로딩 (Milvus 등에서 사용)
# -----------------------------------------------------------------------------
def get_or_load_hf_embedder(model_path: str) -> Tuple[AutoTokenizer, AutoModel]:
    """
    HuggingFace AutoModel/Tokenizer 로더.

    manage_vator_DB.py 등의 서버 사이드 검색에서도 재사용할 수 있게 중앙화한다.
    """
    with _CACHE_LOCK:
        if model_path in _TOKENIZER_CACHE and model_path in _MODEL_CACHE:
            return _TOKENIZER_CACHE[model_path], _MODEL_CACHE[model_path]

        logger.info("[HF Embedding] loading model: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        _TOKENIZER_CACHE[model_path] = tokenizer
        _MODEL_CACHE[model_path] = model
        return tokenizer, model


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> List[str]:
    """간단한 토큰 길이 기반 청크 분할(단어 단위)."""
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, max_tokens - overlap)
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

