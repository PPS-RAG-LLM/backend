from functools import lru_cache
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json
from utils import logger, free_torch_memory, load_embedding_model
from errors import NotFoundError

logger = logger(__name__)


# config 기반 경로 헬퍼
def _doc_dirs():
    from config import config as _cfg
    base = _cfg.get("user_documents", {}) or {}
    return Path(base.get("doc_info_dir", "storage/documents/documents-info")), \
           Path(base.get("vector_cache_dir", "storage/documents/vector-cache"))

# documents-info/<name>.json에서 id 우선, 실패 시 파일명에서 uuid 폴백
def extract_doc_ids_from_attachments(attachments: List[Dict[str, Any]]) -> List[str]:
    """attachments의 name 또는 contentString에서 '-<uuid>.json'을 파싱해 doc_id 목록 반환."""
    logger.info(f"\n\n[extract_doc_ids_from_attachments] \n\n{attachments}\n\n")
    doc_info_dir, _ = _doc_dirs()
    doc_ids: List[str] = []

    for att in attachments or []:
        # dict / pydantic model 모두 지원
        if isinstance(att, dict):
            name = (att.get("name") or att.get("contentString") or "").strip()
        else:
            name = (getattr(att, "name", None) or getattr(att, "contentString", "") or "").strip()

        # URL/경로 섞인 경우 basename만 취함 + 쿼리 제거
        name = name.split("/")[-1].split("?")[0]

        if not name.endswith(".json"):
            continue

        # 1) 파일명에서 UUID 폴백
        base = name[:-5]  # .json 제거
        maybe_uuid = base.rsplit("-", 1)[-1].strip()
        if maybe_uuid and maybe_uuid.count("-") == 4:
            doc_ids.append(maybe_uuid)
            continue

        # 2) 문서 info JSON에서 id 읽기(파일이 있을 경우)
        p = doc_info_dir / name
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                _id = str(j.get("id") or "").strip()
                if _id:
                    doc_ids.append(_id)
            except Exception:
                pass

    # 중복 제거, 순서 유지
    return list(dict.fromkeys(doc_ids))


def _embed_text_local(text: str):
    logger.info(f"embed text {text}")
    m = load_embedding_model()
    result =  m.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    free_torch_memory()
    return result

def _cosine(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float((a @ b) / (na * nb))

def _load_vectors_for_doc(doc_id: str) -> List[Dict[str, Any]]:
    _, vec_dir = _doc_dirs()
    path = vec_dir / f"{doc_id}.json"
    logger.info(f"load vectors for doc {doc_id} from {path}")
    if not path.exists():
        return[]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"load vectors for doc {doc_id} failed: {e}")
        return []

def retrieve_contexts_local(query: str, candidate_doc_ids: List[str], top_k: int, threshold: float) -> List[Dict[str, Any]]:
    qv = _embed_text_local(query)
    hits = []
    for doc_id in candidate_doc_ids or []:
        for it in _load_vectors_for_doc(doc_id):
            vec = it.get("values")
            meta = it.get("metadata") or {}
            if not isinstance(vec, list): continue
            try: v = np.asarray(vec, dtype=float)
            except Exception: continue
            score = _cosine(qv, v)
            if score >= threshold:
                hits.append({"score": score, "doc_id": doc_id, "text": str(meta.get("text") or "")})
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits[: max(1, int(top_k))]

def build_context_message(snippets: List[Dict[str, Any]]) -> str:
    if not snippets: return ""
    parts = [f"[{i}] {h['text']}" for i, h in enumerate(snippets, 1)]
    return "아래 CONTEXTS 를 근거로 한국어로 답변하세요.\n\n### CONTEXTS\n" + "\n---\n".join(parts)