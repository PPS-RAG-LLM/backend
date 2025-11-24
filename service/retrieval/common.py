"""공통 검색/임베딩 유틸리티."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

from config import config as app_config
from utils import free_torch_memory, load_embedding_model, logger

logger = logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _resolve_path(value: Optional[str], fallback: str) -> Path:
    path_val = value or fallback
    path = Path(path_val)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

# -----------------------------------------------------------------------------
# 경로/캐시 헬퍼
# -----------------------------------------------------------------------------
_DOC_INFO_DIR: Optional[Path] = None
_VECTOR_CACHE_DIR: Optional[Path] = None

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}
_MODEL_CACHE: Dict[str, AutoModel] = {}
_HF_DEVICE_CACHE: Dict[str, torch.device] = {}
_CACHE_LOCK = threading.Lock()


def _doc_dirs() -> Tuple[Path, Path]:
    """config.yaml 기반 문서 메타/벡터 디렉토리 경로 반환."""
    global _DOC_INFO_DIR, _VECTOR_CACHE_DIR
    if _DOC_INFO_DIR is not None and _VECTOR_CACHE_DIR is not None:
        return _DOC_INFO_DIR, _VECTOR_CACHE_DIR
    doc_cfg = app_config.get("user_documents", {}) or {}
    _DOC_INFO_DIR = _resolve_path(doc_cfg.get("doc_info_dir"), "storage/documents/documents-info")
    _VECTOR_CACHE_DIR = _resolve_path(doc_cfg.get("vector_cache_dir"), "storage/documents/vector-cache")
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
def get_or_load_hf_embedder(model_path: str) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """
    HuggingFace AutoModel/Tokenizer 로더.

    manage_vator_DB.py 등의 서버 사이드 검색에서도 재사용할 수 있게 중앙화한다.
    """
    with _CACHE_LOCK:
        if (
            model_path in _TOKENIZER_CACHE
            and model_path in _MODEL_CACHE
            and model_path in _HF_DEVICE_CACHE
        ):
            return (
                _TOKENIZER_CACHE[model_path],
                _MODEL_CACHE[model_path],
                _HF_DEVICE_CACHE[model_path],
            )

        logger.info("[HF Embedding] loading model: %s", model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        model.eval()

        _TOKENIZER_CACHE[model_path] = tokenizer
        _MODEL_CACHE[model_path] = model
        _HF_DEVICE_CACHE[model_path] = device
        return tokenizer, model, device


def hf_mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def hf_embed_text(tok, model, device, text: str, max_len: int = 512) -> np.ndarray:
    inputs = tok(
        text,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outs = model(**inputs)
    vec = (
        hf_mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
    )
    return vec


def chunk_text(
    text: str,
    max_tokens: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """간단한 토큰 길이 기반 청크 분할(단어 단위)."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must satisfy 0 <= overlap < max_tokens")
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

def split_for_varchar_bytes(
    text: str,
    hard_max_bytes: int = 32768,
    soft_max_bytes: int = 30000,   # 여유 버퍼
    table_mark: str = "[[TABLE",
) -> list[str]:
    """
    VARCHAR 초과 방지: UTF-8 바이트 기준으로 안전 분할.
    - 표 텍스트는 헤더([[TABLE ...]])를 첫 조각에만 포함.
    - 이후 조각엔 [[TABLE_CONT i/n]] 마커를 부여.
    - 개행 경계 우선(backtrack), 그래도 안되면 하드컷.
    """
    if not text:
        return [""]

    # 표 헤더 분리
    header = ""
    body = text
    if text.startswith(table_mark):
        head_end = text.find("]]")
        if head_end != -1:
            head_end += 2
            if head_end < len(text) and text[head_end] == "\n":
                head_end += 1
            header, body = text[:head_end], text[head_end:]

    def _split_body(b: str) -> list[str]:
        out: list[str] = []
        b_bytes = b.encode("utf-8")
        n = len(b_bytes)
        i = 0
        while i < n:
            j = min(i + soft_max_bytes, n)
            # 개행 경계로 뒤로 물러나기
            k = j
            backtracked = False
            # j부터 i까지 역방향으로 \n 바이트(0x0A) 탐색
            while k > i and (j - k) < 2000:  # 최대 2KB만 백트랙
                if b_bytes[k-1:k] == b"\n":
                    backtracked = True
                    break
                k -= 1
            if backtracked and (k - i) >= int(soft_max_bytes * 0.6):
                cut = k
            else:
                cut = j

            # 하드 컷(멀티바이트 경계 맞추기)
            if cut - i > hard_max_bytes:
                cut = i + hard_max_bytes

            # UTF-8 안전 디코드: 경계가 문자를 반쯤 자를 수 있으니 넉넉히 조정
            chunk = b_bytes[i:cut]
            # 만약 디코드 에러가 나면 한 바이트씩 줄이며 안전 경계 찾기
            while True:
                try:
                    s = chunk.decode("utf-8")
                    break
                except UnicodeDecodeError:
                    cut -= 1
                    if cut <= i:
                        # 최악의 경우 한 글자라도 디코드되게 한 바이트 앞당김
                        cut = i + 1
                    chunk = b_bytes[i:cut]
            out.append(s)
            i = cut
        return out

    if len(text.encode("utf-8")) <= hard_max_bytes:
        return [text]

    parts = _split_body(body)
    if header:
        total = len(parts)
        result = []
        for idx, c in enumerate(parts, start=1):
            if idx == 1:
                # 첫 조각은 헤더 + 본문
                # 전체가 하드맥스를 넘지 않게 헤더와 합친 뒤 한번 더 자르기
                first = header + c
                if len(first.encode("utf-8")) <= hard_max_bytes:
                    result.append(first)
                else:
                    # 너무 크면 헤더는 유지하고 c를 다시 잘라 붙임
                    # (헤더가 길 때 매우 예외적)
                    subparts = _split_body(c)
                    if subparts:
                        # 첫 조각은 헤더 + 첫 sub
                        f = header + subparts[0]
                        if len(f.encode("utf-8")) > hard_max_bytes:
                            # 헤더 자체가 큰 극단: 헤더만 넣고 이후 CONT로 처리
                            result.append(header[:0] + header)  # 그대로
                            # 나머지는 CONT
                            for sidx, sp in enumerate(subparts, start=1):
                                tag = f"[[TABLE_CONT {sidx}/{len(subparts)}]]\n"
                                result.append(tag + sp)
                        else:
                            result.append(f)
                            # 나머지는 CONT
                            for sidx, sp in enumerate(subparts[1:], start=2):
                                tag = f"[[TABLE_CONT {sidx}/{len(subparts)}]]\n"
                                result.append(tag + sp)
                    else:
                        result.append(header)  # 본문이 없으면 헤더만
            else:
                tag = f"[[TABLE_CONT {idx}/{total}]]\n"
                # tag + c 가 하드맥스를 넘지 않도록 재자르기
                rest = tag + c
                if len(rest.encode("utf-8")) <= hard_max_bytes:
                    result.append(rest)
                else:
                    subs = _split_body(c)
                    for sidx, sp in enumerate(subs, start=1):
                        subt = f"[[TABLE_CONT {idx}.{sidx}/{total}]]\n" + sp
                        if len(subt.encode("utf-8")) <= hard_max_bytes:
                            result.append(subt)
                        else:
                            # 그래도 넘으면 하드컷으로 마지막 방어
                            bb = subt.encode("utf-8")[:hard_max_bytes]
                            result.append(bb.decode("utf-8", errors="ignore"))
        return result
    else:
        return parts

def parse_doc_version(stem: str) -> Tuple[str, int]:
    """
    파일명(stem)에서 버전(타임스탬프 등)을 분리.
    예: "doc_20231025" -> ("doc", 20231025)
    """
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0

def determine_level_for_task(text: str, task_rules: Dict) -> int:
    """
    텍스트 내 키워드를 기반으로 보안 레벨 결정.
    task_rules: {"maxLevel": N, "levels": {"1": [...], "2": [...]}}
    """
    max_level = int(task_rules.get("maxLevel", 1))
    levels = task_rules.get("levels", {})
    sel = 1
    # 상위 레벨 우선
    for lvl in range(1, max_level + 1):
        kws = levels.get(str(lvl), [])
        for kw in kws:
            if kw and kw in text:
                sel = max(sel, lvl)
    return sel

def extract_insert_ids(result: Any) -> List[str]:
    """
    Milvus insert 결과에서 primary key 리스트를 추출한다.
    다양한 리턴 타입(dict/InsertResult 등)을 모두 처리한다.
    """
    ids: Any = None
    if isinstance(result, dict):
        ids = (
            result.get("ids")
            or result.get("primary_keys")
            or result.get("inserted_ids")
        )
    else:
        ids = getattr(result, "primary_keys", None) or getattr(result, "ids", None)
    if not ids:
        return []
    return [str(pk) for pk in ids]

