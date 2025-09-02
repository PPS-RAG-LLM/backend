# === Vector DB Service (Milvus Lite) ===
# NOTE: Migrated from experimental src/v1/main.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from collections import defaultdict
import logging
import sqlite3
import shutil

import torch
from pydantic import BaseModel
from pymilvus import MilvusClient, DataType
from transformers import AutoModel, AutoTokenizer

__all__ = [
    "PDFExtractRequest",
    "RAGSearchRequest",
    "SinglePDFIngestRequest",
    "extract_pdfs",
    "ingest_embeddings",
    "ingest_single_pdf",
    "search_documents",
    "delete_db",
]

# -------------------------------------------------
# Paths & Constants
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # backend/service/admin
resource_dir = Path(os.getenv("RESOURCE_DIR", BASE_DIR / "resources")).resolve()

EXTRACTED_TEXT_DIR = resource_dir / "extracted_texts"
META_JSON_PATH = EXTRACTED_TEXT_DIR / "_extraction_meta.json"
 
# Model root directory
MODEL_ROOT_DIR = (resource_dir / "model").resolve()

MILVUS_LITE_PATH = (resource_dir / "milvus_lite.db").resolve()
COLLECTION_NAME = "pdf_chunks"

# Fixed project/data paths
PROJECT_ROOT = BASE_DIR.parent.parent  # backend/
RAW_DATA_DIR = (PROJECT_ROOT / "storage" / "user_data" / "row_data").resolve()
LOCAL_DATA_ROOT = (PROJECT_ROOT / "storage" / "user_data" / "local_data").resolve()
SQLITE_DB_PATH = (PROJECT_ROOT / "storage" / "pps_rag.db").resolve()

logger = logging.getLogger(__name__)

# -------------------------------------------------
# Pydantic Request Schemas
# -------------------------------------------------
class PDFExtractRequest(BaseModel):
    """Deprecated: kept for compatibility; path is now fixed to ./storage/user_data/local_data."""
    pass


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_level: int = 1
    model: str | None = "bge"  # kept for backward-compat; API now uses saved settings


class SinglePDFIngestRequest(BaseModel):
    pdf_path: str
    model: str | None = "bge"  # 'bge' | 'qwen'
    workspace_id: int | None = None


# === Runtime-configurable Defaults ===
_CURRENT_EMBED_MODEL_KEY = "bge"   # 'bge' | 'qwen' ...
_CURRENT_SEARCH_TYPE = "hybrid"     # 'hybrid' | 'bm25'


def set_vector_settings(embed_model_key: str | None = None, search_type: str | None = None):
    """Update runtime defaults for embedding model & search type."""
    global _CURRENT_EMBED_MODEL_KEY, _CURRENT_SEARCH_TYPE  # noqa: PLW0603

    if embed_model_key is not None:
        # Resolve flexibly against available local model directories
        canonical, _ = _resolve_model_input(embed_model_key)
        _CURRENT_EMBED_MODEL_KEY = canonical

    if search_type is not None:
        st = search_type.lower()
        if st not in {"hybrid", "bm25"}:
            raise ValueError("unsupported searchType; allowed: 'hybrid', 'bm25'")
        _CURRENT_SEARCH_TYPE = st


def get_vector_settings():
    return {
        "embeddingModel": _CURRENT_EMBED_MODEL_KEY,
        "searchType": _CURRENT_SEARCH_TYPE,
    }


# -------------------------------------------------
# Embedding utilities
# -------------------------------------------------

def _mean_pooling(outputs, mask):  # type: ignore[valid-type]
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _resolve_model_input(model_key: str | None) -> tuple[str, Path]:
    """Resolve input to a concrete embedding model directory.

    Accepts flexible keys, e.g., 'bge', 'bge_m3', 'embedding_bge_m3',
    'qwen', 'qwen3_4b', 'qwen3_0_6b', 'embedding_qwen3_4b'.
    Returns (canonical_key, model_dir_path). canonical_key is the directory name.
    """
    models: list[Path] = []
    if MODEL_ROOT_DIR.exists():
        for child in MODEL_ROOT_DIR.iterdir():
            if child.is_dir():
                models.append(child.resolve())

    key = (model_key or "bge").lower()

    def aliases(p: Path) -> list[str]:
        nm = p.name.lower()
        als = [nm]
        if nm.startswith("embedding_"):
            als.append(nm[len("embedding_"):])
        return als

    # 1) Exact/alias match
    for p in models:
        als = aliases(p)
        if key in als:
            return p.name, p

    # 2) Substring match
    for p in models:
        if key in p.name.lower():
            return p.name, p

    # 3) Heuristic by family
    if "qwen" in key:
        # prefer larger variants if multiple exist
        preferred = [m for m in models if "qwen" in m.name.lower()]
        if preferred:
            # sort by name length desc as crude preference
            preferred.sort(key=lambda x: len(x.name), reverse=True)
            p = preferred[0]
            return p.name, p

    # default to bge family
    preferred = [m for m in models if "bge" in m.name.lower()]
    if preferred:
        p = preferred[0]
        return p.name, p

    # Fallback to conventional path even if not present (will error later)
    fallback = MODEL_ROOT_DIR / "embedding_bge_m3"
    return fallback.name, fallback


def _get_model_dir(model_key: str | None) -> Path:
    return _resolve_model_input(model_key)[1]


def _load_embedder(model_key: str | None = "bge"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = _get_model_dir(model_key)
    if not model_dir.exists():
        raise FileNotFoundError(f"[Embedding Model] 경로가 없습니다: {model_dir}")

    need_files = [
        model_dir / "tokenizer_config.json",
        model_dir / "tokenizer.json",
        model_dir / "config.json",
    ]
    if not all(p.exists() for p in need_files):
        raise FileNotFoundError(f"[Embedding Model] 필수 파일 누락: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        .to(device)
        .eval()
    )
    return tokenizer, model, device


def _embed_text(tokenizer, model, device, text: str, max_len: int = 512):  # type: ignore[valid-type]
    inputs = tokenizer(text, truncation=True, padding="longest", max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        outs = model(**inputs)
    vec = _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
    return vec


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _parse_doc_version(stem: str):
    """Return (doc_id, version) by parsing file stem."""
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0


def _client() -> MilvusClient:
    MILVUS_LITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return MilvusClient(str(MILVUS_LITE_PATH))


# -------------------------------------------------
# SQLite helpers for Security Level Rules
# -------------------------------------------------

def _db_connect() -> sqlite3.Connection:
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS security_level_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            max_level INTEGER NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS security_level_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level INTEGER NOT NULL,
            keyword TEXT NOT NULL
        )
        """
    )
    return conn


def _parse_at_string_to_keywords(value: str) -> list[str]:
    # Split by '@' and strip; filter empties
    if not value:
        return []
    tokens = [tok.strip() for tok in value.split("@")]
    return [t for t in tokens if t]


def set_security_level_rules(max_level: int, levels_map: dict[int, str]) -> dict:
    """Persist security rules.

    levels_map maps level -> '@' delimited string (e.g., "@출장@결혼").
    """
    if max_level < 1:
        raise ValueError("max_level must be >= 1")

    # Disallow configuring level 1 keywords
    for level, value in levels_map.items():
        if int(level) == 1 and _parse_at_string_to_keywords(value):
            raise ValueError("level 1 cannot have keywords; it is always accessible to all")

    conn = _db_connect()
    try:
        with conn:
            # Upsert config (single row id=1)
            conn.execute(
                "INSERT INTO security_level_config(id, max_level) VALUES(1, ?) "
                "ON CONFLICT(id) DO UPDATE SET max_level=excluded.max_level, updated_at=CURRENT_TIMESTAMP",
                (int(max_level),),
            )
            # Clear and insert keywords
            conn.execute("DELETE FROM security_level_keywords")
            for level, value in levels_map.items():
                try:
                    lvl = int(level)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Invalid level key '%s' in levels_map", level)
                    continue
                if lvl < 1 or lvl > max_level:
                    continue
                if lvl == 1:
                    # Do not store any keywords for level 1
                    continue
                for kw in _parse_at_string_to_keywords(value):
                    conn.execute(
                        "INSERT INTO security_level_keywords(level, keyword) VALUES(?, ?)",
                        (lvl, kw),
                    )
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to set security level rules")
        raise
    finally:
        conn.close()

    return get_security_level_rules()


def get_security_level_rules() -> dict:
    conn = _db_connect()
    try:
        cur = conn.cursor()
        row = cur.execute("SELECT max_level FROM security_level_config WHERE id=1").fetchone()
        max_level = int(row[0]) if row else 1
        rows = cur.execute(
            "SELECT level, keyword FROM security_level_keywords ORDER BY level ASC, keyword ASC"
        ).fetchall()
        levels: dict[int, list[str]] = {i: [] for i in range(1, max_level + 1)}
        for lvl, kw in rows:
            if int(lvl) not in levels:
                levels[int(lvl)] = []
            levels[int(lvl)].append(str(kw))
        return {
            "maxLevel": max_level,
            "levels": {str(k): v for k, v in levels.items()},
        }
    except Exception:  # noqa: BLE001
        logger.exception("Failed to get security level rules")
        raise
    finally:
        conn.close()


def _determine_security_level_from_text(text: str, rules: dict) -> int:
    """Return highest level whose any keyword appears in text. Defaults to 1."""
    try:
        max_level = int(rules.get("maxLevel", 1))
        levels = rules.get("levels", {})
        selected = 1
        # Simple substring matching; prioritize higher levels
        for lvl in range(1, max_level + 1):
            keywords = levels.get(str(lvl), [])
            for kw in keywords:
                if kw and kw in text:
                    if lvl > selected:
                        selected = lvl
        return selected
    except Exception:  # noqa: BLE001
        logger.exception("Error determining security level from text")
        return 1


def _ensure_collection_and_index(client: MilvusClient, emb_dim: int, metric: str = "IP"):
    cols = client.list_collections()
    if COLLECTION_NAME not in cols:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False, description="PDF Chunk Storage")
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
        schema.add_field("path", DataType.VARCHAR, max_length=500)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("security_level", DataType.INT64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("version", DataType.INT64)
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

    try:
        idx_list = client.list_indexes(collection_name=COLLECTION_NAME, field_name="embedding")
    except Exception:  # noqa: BLE001
        idx_list = []
    if not idx_list:
        ip = client.prepare_index_params()
        ip.add_index("embedding", "FLAT", metric_type=metric, params={})
        client.create_index(COLLECTION_NAME, ip, timeout=120.0, sync=True)

    try:
        client.load_collection(collection_name=COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass


# -------------------------------------------------
# 1) PDF → 텍스트 추출 (fixed paths + security rules)
# -------------------------------------------------
async def extract_pdfs():
    import fitz  # type: ignore
    from tqdm import tqdm  # type: ignore

    # Ensure base directories
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare security folders based on current rules
    rules = get_security_level_rules()
    max_level = int(rules.get("maxLevel", 1))
    for lvl in range(1, max_level + 1):
        (LOCAL_DATA_ROOT / f"securityLevel{lvl}").mkdir(parents=True, exist_ok=True)

    # Load previous meta if any
    done_files: dict[str, dict] = {}
    if META_JSON_PATH.exists():
        try:
            done_files = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read extraction meta JSON")
            done_files = {}

    new_meta: dict[str, dict] = {}
    pdf_paths = list(RAW_DATA_DIR.rglob("*.pdf"))
    # Deduplicate by keeping only the latest version inferred from filename pattern with '_' and date-like suffix
    # Example: "28._연봉제시행규정_20191128.pdf" => base key: "연봉제시행규정", date: 20191128
    def _extract_base_and_date(p: Path):
        name = p.stem  # without .pdf
        parts = name.split("_")
        # take last as date candidate if all digits and length in {4,6,8}
        date_num = 0
        if len(parts) >= 2:
            cand = parts[-1]
            if cand.isdigit() and len(cand) in (4, 6, 8):
                try:
                    date_num = int(cand)
                except Exception:  # noqa: BLE001
                    date_num = 0
        # base: skip numeric prefixes and empty tokens, take the longest mid token
        mid_tokens = [t for t in parts[:-1] if t and not t.isdigit()]
        base = max(mid_tokens, key=len) if mid_tokens else parts[0]
        return base, date_num

    # Group by base and choose latest; remove older duplicates physically
    grouped: dict[str, list[tuple[Path, int]]] = {}
    for p in pdf_paths:
        base, date_num = _extract_base_and_date(p)
        grouped.setdefault(base, []).append((p, date_num))

    kept_paths: list[Path] = []
    removed_files: list[str] = []
    for base, lst in grouped.items():
        # pick max date_num; if all zero, keep the longest filename to be deterministic
        lst_sorted = sorted(lst, key=lambda x: (x[1], len(x[0].name)))
        keep = lst_sorted[-1]
        kept_paths.append(keep[0])
        to_remove = [p for (p, d) in lst_sorted[:-1]]
        for old_path in to_remove:
            try:
                old_path.unlink(missing_ok=True)
                removed_files.append(str(old_path.relative_to(RAW_DATA_DIR)))
            except Exception:  # noqa: BLE001
                logger.exception("Failed to remove duplicate file: %s", old_path)
    # Use only latest files
    pdf_paths = kept_paths
    if not pdf_paths:
        return {"message": "처리할 PDF가 없습니다.", "meta_path": str(META_JSON_PATH), "deduplicated": {"removedCount": len(removed_files), "removed": removed_files}}

    for pdf_path in tqdm(pdf_paths, desc="PDF 전처리"):
        try:
            # Extract text to determine level
            doc = fitz.open(pdf_path)
            text_pages = [p.get_text("text").strip() for p in doc]
            pdf_text = "\n\n".join(text_pages)

            # Determine level from rules
            sec_level = _determine_security_level_from_text(pdf_text, rules)
            sec_folder = f"securityLevel{sec_level}"

            # Destination relative path inside local_data/securityLevelN
            rel_from_raw = pdf_path.relative_to(RAW_DATA_DIR)
            dest_rel_pdf = Path(sec_folder) / rel_from_raw

            # Copy PDF into local_data structure
            dest_pdf_abs = LOCAL_DATA_ROOT / dest_rel_pdf
            dest_pdf_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(pdf_path, dest_pdf_abs)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to copy PDF to local_data: %s", dest_pdf_abs)

            # Write extracted text under extracted_texts mirroring the same rel path
            txt_path = EXTRACTED_TEXT_DIR / dest_rel_pdf.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # Meta info
            doc_id_part, version_num = _parse_doc_version(rel_from_raw.stem)
            lines = pdf_text.splitlines()
            key = str(dest_rel_pdf)
            info = {
                "chars": len(pdf_text),
                "lines": len(lines),
                "preview": pdf_text[:200].replace("\n", " ") + "…",
                "security_level": int(sec_level),
                "doc_id": doc_id_part,
                "version": version_num,
            }
            new_meta[key] = info
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to process PDF: %s", pdf_path)
            # Keep an error entry in meta with destination under level 1 by default
            try:
                rel_from_raw = pdf_path.relative_to(RAW_DATA_DIR)
                dest_rel_pdf = Path("securityLevel1") / rel_from_raw
                new_meta[str(dest_rel_pdf)] = {"error": str(e)}
            except Exception:
                new_meta[str(pdf_path.name)] = {"error": str(e)}

    META_JSON_PATH.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "message": "PDF 추출 완료",
        "pdf_count": len(pdf_paths),
        "meta_path": str(META_JSON_PATH),
        "deduplicated": {
            "removedCount": len(removed_files),
            "removed": removed_files,
        },
    }


# -------------------------------------------------
# 2) 전체 임베딩 & 인제스트
# -------------------------------------------------
async def ingest_embeddings(model_key: str | None = None, chunk_size: int | None = None, overlap: int | None = None):
    if model_key is None:
        model_key = _CURRENT_EMBED_MODEL_KEY
    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}

    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    tokenizer, model, device = _load_embedder(model_key)
    emb_dim = int(_embed_text(tokenizer, model, device, "test").shape[0])

    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    # Chunking parameters (with sane bounds)
    MAX_TOKENS = 512
    if isinstance(chunk_size, int) and chunk_size > 0:
        MAX_TOKENS = int(chunk_size)
    OVERLAP = 64
    if isinstance(overlap, int) and overlap >= 0:
        OVERLAP = max(0, min(int(overlap), max(0, MAX_TOKENS - 1)))

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    total_inserted = 0

    for txt_path in EXTRACTED_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(EXTRACTED_TEXT_DIR)
        rel_pdf = rel_txt.with_suffix(".pdf").as_posix()
        if rel_pdf not in extraction_meta:
            continue

        meta_entry = extraction_meta[rel_pdf]
        sec_level = meta_entry.get("security_level", 1)
        doc_id = meta_entry.get("doc_id")
        version = meta_entry.get("version", 0)

        if not doc_id or version == 0:
            _id_part, _ver_num = _parse_doc_version(rel_txt.stem)
            if not doc_id:
                doc_id = _id_part
                meta_entry["doc_id"] = doc_id
            if version == 0:
                version = _ver_num
                meta_entry["version"] = version

        try:
            client.delete(COLLECTION_NAME, filter=f"doc_id == '{doc_id}' && version <= {version}")
        except Exception:  # noqa: BLE001
            pass

        text = txt_path.read_text(encoding="utf-8")
        rows = []
        for idx, chunk in enumerate(chunk_text(text)):
            vec = _embed_text(tokenizer, model, device, chunk, max_len=MAX_TOKENS)
            rows.append(
                {
                    "embedding": vec.tolist(),
                    "path": str(rel_txt),
                    "chunk_idx": int(idx),
                    "security_level": int(sec_level),
                    "doc_id": str(doc_id),
                    "version": int(version),
                }
            )
            if len(rows) >= 128:
                client.insert(COLLECTION_NAME, rows)
                total_inserted += len(rows)
                rows = []
        if rows:
            client.insert(COLLECTION_NAME, rows)
            total_inserted += len(rows)

    META_JSON_PATH.write_text(json.dumps(extraction_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        client.flush(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")
    return {"message": "Ingest 완료(Milvus Lite)", "inserted_chunks": total_inserted}


# -------------------------------------------------
# 2-1) 단일 PDF 인제스트
# -------------------------------------------------
async def ingest_single_pdf(req: SinglePDFIngestRequest):
    import fitz  # pylint: disable=import-error
    from repository.documents import insert_workspace_document
    import uuid

    if META_JSON_PATH.exists():
        extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    else:
        extraction_meta = {}

    pdf_path = Path(req.pdf_path)
    if not pdf_path.exists():
        return {"error": f"PDF 경로를 찾을 수 없습니다: {pdf_path}"}

    def _ensure_single_extracted(pdf_abs: Path):
        try:
            lvl_folder = next(p for p in pdf_abs.parents if p.name.startswith("securityLevel"))
            sec_level_val = int(lvl_folder.name.replace("securityLevel", ""))
        except StopIteration:
            sec_level_val = 1

        doc = fitz.open(pdf_abs)
        text_all = "\n\n".join(p.get_text("text").strip() for p in doc)

        root_local = Path("local_data")
        try:
            rel_pdf = pdf_abs.relative_to(root_local)
        except ValueError:
            rel_pdf = Path(pdf_abs.name)

        txt_path_local = EXTRACTED_TEXT_DIR / rel_pdf.with_suffix(".txt")
        txt_path_local.parent.mkdir(parents=True, exist_ok=True)
        txt_path_local.write_text(text_all, encoding="utf-8")

        stem = rel_pdf.stem
        doc_id_part, ver_num = _parse_doc_version(stem)

        extraction_meta[str(rel_pdf)] = {
            "chars": len(text_all),
            "lines": len(text_all.splitlines()),
            "preview": text_all[:200].replace("\n", " ") + "…",
            "security_level": sec_level_val,
            "doc_id": doc_id_part,
            "version": ver_num,
        }
        META_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_JSON_PATH.write_text(json.dumps(extraction_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(rel_pdf)

    meta_key = next((k for k in extraction_meta if k.endswith(pdf_path.name)), None)
    if meta_key is None:
        meta_key = _ensure_single_extracted(pdf_path)
        extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    txt_path = EXTRACTED_TEXT_DIR / Path(meta_key).with_suffix(".txt")
    if not txt_path.exists():
        return {"error": f"텍스트 파일이 존재하지 않습니다: {txt_path}"}

    meta_entry = extraction_meta[meta_key]
    sec_level = meta_entry["security_level"]
    doc_id = meta_entry.get("doc_id")
    version = meta_entry.get("version", 0)

    # If this is a user/workspace upload, allow duplicates by assigning a unique doc_id
    if req.workspace_id is not None:
        base_stem = Path(meta_key).stem
        unique_suffix = uuid.uuid4().hex[:8]
        doc_id = f"u{int(req.workspace_id)}__{base_stem}__{unique_suffix}"

    # 임베더/클라
    # Always use saved embedding model from settings if available
    try:
        saved_model_key = get_vector_settings()["embeddingModel"]
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load vector settings; defaulting to 'bge'")
        saved_model_key = "bge"
    tokenizer, model, device = _load_embedder(saved_model_key)
    emb_dim = int(_embed_text(tokenizer, model, device, "test").shape[0])
    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    # 구버전 삭제 (관리자/글로벌 문서에만 적용; 사용자 업로드는 고유 doc_id 이므로 스킵 안전)
    try:
        client.delete(COLLECTION_NAME, filter=f"doc_id == '{doc_id}' && version <= {version}")
    except Exception:
        pass

    # 청크 & 삽입
    MAX_TOKENS, OVERLAP = 512, 64

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    text = txt_path.read_text(encoding="utf-8")
    rows, cnt = [], 0
    for idx, chunk in enumerate(chunk_text(text)):
        vec = _embed_text(tokenizer, model, device, chunk, max_len=MAX_TOKENS)
        rows.append(
            {
                "embedding": vec.tolist(),
                "path": str(Path(meta_key).with_suffix(".txt")),
                "chunk_idx": int(idx),
                "security_level": int(sec_level),
                "doc_id": str(doc_id),
                "version": int(version),
            }
        )
        if len(rows) >= 128:
            client.insert(COLLECTION_NAME, rows)
            cnt += len(rows)
            rows = []
    if rows:
        client.insert(COLLECTION_NAME, rows)
        cnt += len(rows)

    # Record into SQL if workspace_id provided
    if req.workspace_id is not None:
        try:
            insert_workspace_document(
                doc_id=str(doc_id),
                filename=Path(meta_key).with_suffix(".pdf").name,
                docpath=str(Path(meta_key).with_suffix(".pdf")),
                workspace_id=int(req.workspace_id),
                metadata={
                    "securityLevel": int(sec_level),
                    "model": (req.model or "bge"),
                    "chunks": int(cnt),
                    "isUserUpload": True,
                    "baseDocId": Path(meta_key).stem,
                },
            )
        except Exception:
            pass

    # 마무리: flush + 인덱스/로드 재보장
    try:
        client.flush(COLLECTION_NAME)
    except Exception:  # noqa: BLE001
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    return {
        "message": "단일 PDF 인제스트 완료(Milvus Lite)",
        "doc_id": doc_id,
        "version": version,
        "chunks": cnt,
    }


# -------------------------------------------------
# 3) 검색
# -------------------------------------------------
async def search_documents(req: RAGSearchRequest):
    start_time = time.perf_counter()

    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다."}

    tokenizer, model, device = _load_embedder(req.model or "bge")
    q_emb = _embed_text(tokenizer, model, device, req.query)
    client = _client()

    _ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP")
    if COLLECTION_NAME not in client.list_collections():
        return {"error": "컬렉션이 없습니다. 먼저 인제스트를 수행하세요."}

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[q_emb.tolist()],
        anns_field="embedding",
        limit=int(req.top_k),
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["path", "chunk_idx", "security_level"],
        filter=f"security_level <= {int(req.user_level)}",
    )

    MAX_TOKENS, OVERLAP = 512, 64

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunks.append(" ".join(words[start:end]).strip())
            start += max_tokens - overlap
        return [c for c in chunks if c]

    hits = []
    for hit in results[0]:
        if isinstance(hit, dict):
            ent = hit.get("entity", {})
            path = ent.get("path")
            cidx = ent.get("chunk_idx")
            sec_level = ent.get("security_level")
            score = hit.get("distance")
        else:
            path = hit.entity.get("path")
            cidx = hit.entity.get("chunk_idx")
            sec_level = hit.entity.get("security_level")
            score = hit.score

        try:
            full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to read text for path: %s", path)
            full_txt = ""
        chunks = chunk_text(full_txt)
        idx = int(cidx) if isinstance(cidx, int) else int(cidx or 0)
        if not chunks:
            snippet = ""
        elif idx < 0:
            snippet = chunks[0]
        elif idx >= len(chunks):
            snippet = chunks[-1]
        else:
            snippet = chunks[idx]
        hits.append(
            {
                "score": float(score),
                "path": path,
                "chunk_idx": int(cidx),
                "security_level": int(sec_level),
                "snippet": snippet,
            }
        )

    context = "\n---\n".join(h["snippet"] for h in hits)
    prompt = f"사용자 질의: {req.query}\n\n관련 문서 스니펫:\n{context}\n\n위 내용을 참고하여 응답해 주세요."
    elapsed = round(time.perf_counter() - start_time, 4)
    return {"elapsed_sec": elapsed, "hits": hits, "prompt": prompt}


async def execute_search(question: str, top_k: int = 5, security_level: int = 1, source_filter: list[str] | None = None, model_key: str | None = None):
    req = RAGSearchRequest(query=question, top_k=top_k, user_level=security_level, model=model_key)
    res = await search_documents(req)
    if source_filter:
        names = {Path(n).stem for n in source_filter}
        filtered_hits = [h for h in res.get("hits", []) if Path(h["path"]).stem in names]
        res["hits"] = filtered_hits
    return res


# -------------------------------------------------
# 4) 컬렉션 삭제
# -------------------------------------------------
async def delete_db():
    client = _client()
    cols = client.list_collections()
    for col in cols:
        client.drop_collection(col)
    return {"message": "삭제 완료(Milvus Lite)", "dropped_collections": cols}


async def list_indexed_files(limit: int = 16384, offset: int = 0, query: str | None = None):
    """Return aggregated file entries. Supports pagination via limit/offset.
    If query is provided, filters by fileName contains query (case-sensitive substring).
    """
    limit = max(1, min(limit, 16384))
    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        return []

    try:
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter="",
            output_fields=["path", "chunk_idx", "security_level"],
            limit=limit,
            offset=offset,
        )
    except Exception:
        rows = []

    # Group by path
    grouped: dict[str, dict] = {}
    counts: dict[str, int] = defaultdict(int)
    levels: dict[str, int] = {}

    for r in rows:
        path = r.get("path") if isinstance(r, dict) else r["path"]  # type: ignore[index]
        counts[path] += 1
        if path not in levels:
            levels[path] = int(r.get("security_level") if isinstance(r, dict) else r["security_level"])  # type: ignore[index]

    items = []
    for path, cnt in counts.items():
        txt_rel = Path(path)
        pdf_rel = txt_rel.with_suffix(".pdf")
        file_name = pdf_rel.name
        txt_abs = EXTRACTED_TEXT_DIR / txt_rel
        try:
            stat = txt_abs.stat()
            size = stat.st_size
            indexed_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        except FileNotFoundError:
            size = None
            indexed_at = None
        items.append({
            "fileName": file_name,
            "filePath": str(pdf_rel),
            "chunkCount": int(cnt),
            "indexedAt": indexed_at,
            "fileSize": size,
            "securityLevel": int(levels.get(path, 1)),
        })
    if query:
        q = str(query)
        items = [it for it in items if q in it["fileName"]]
    return items


async def delete_files_by_names(file_names: list[str]):
    """Delete all chunks whose doc_id matches any of the given file name stems."""
    if not file_names:
        return {"deleted": 0}

    from repository.documents import delete_workspace_documents_by_filenames

    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        # Still try to delete in SQL
        deleted_sql = delete_workspace_documents_by_filenames(file_names)
        return {"deleted": 0, "deleted_sql": deleted_sql, "requested": len(file_names)}

    deleted_total = 0
    for name in file_names:
        stem = Path(name).stem
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                filter=f"doc_id == '{stem}'",
            )
            deleted_total += 1
        except Exception:
            pass

    deleted_sql = delete_workspace_documents_by_filenames(file_names)
    return {"deleted": deleted_total, "deleted_sql": deleted_sql, "requested": len(file_names)}
