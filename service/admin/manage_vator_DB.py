# === Vector DB Service (Milvus Lite) ===
# NOTE: Migrated from experimental src/v1/main.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

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

BGE_MODEL_DIR = (resource_dir / "model" / "embedding_bge_m3").resolve()
QWEN_MODEL_DIR = (resource_dir / "model" / "embedding_qwen3_4b").resolve()

MILVUS_LITE_PATH = (resource_dir / "milvus_lite.db").resolve()
COLLECTION_NAME = "pdf_chunks"

# -------------------------------------------------
# Pydantic Request Schemas
# -------------------------------------------------
class PDFExtractRequest(BaseModel):
    dir_path: str  # Root directory that contains PDFs


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_level: int = 1
    model: str | None = "bge"  # 'bge' | 'qwen'


class SinglePDFIngestRequest(BaseModel):
    pdf_path: str
    model: str | None = "bge"  # 'bge' | 'qwen'


# -------------------------------------------------
# Embedding utilities
# -------------------------------------------------

def _mean_pooling(outputs, mask):  # type: ignore[valid-type]
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _get_model_dir(model_key: str | None) -> Path:
    key = (model_key or "bge").lower()
    if key.startswith("qwen"):
        return QWEN_MODEL_DIR
    return BGE_MODEL_DIR


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
        client.create_collection(COLLECTION_NAME, schema)

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
# 1) PDF → 텍스트 추출
# -------------------------------------------------
async def extract_pdfs(req: PDFExtractRequest):
    import fitz  # type: ignore
    from tqdm import tqdm  # type: ignore

    root_dir = Path(req.dir_path)
    if not root_dir.exists():
        return {"error": f"경로가 존재하지 않습니다: {req.dir_path}"}

    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    done_files: dict[str, dict] = {}
    if META_JSON_PATH.exists():
        done_files = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    new_meta: dict[str, dict] = {}
    pdf_paths = list(root_dir.rglob("*.pdf"))
    if not pdf_paths:
        return {"message": "처리할 PDF가 없습니다."}

    for pdf_path in tqdm(pdf_paths, desc="PDF 전처리"):
        pdf_rel = pdf_path.relative_to(root_dir)
        txt_path = EXTRACTED_TEXT_DIR / pdf_rel.with_suffix(".txt")
        key = str(pdf_rel)
        if key in done_files and txt_path.exists():
            new_meta[key] = done_files[key]
            continue
        try:
            doc = fitz.open(pdf_path)
            text_pages = [p.get_text("text").strip() for p in doc]
            pdf_text = "\n\n".join(text_pages)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # 보안레벨 추정
            level_folder = pdf_rel.parts[0] if len(pdf_rel.parts) else "securityLevel1"
            try:
                security_level = int(level_folder.replace("securityLevel", ""))
            except ValueError:
                security_level = 1

            doc_id_part, version_num = _parse_doc_version(pdf_rel.stem)

            lines = pdf_text.splitlines()
            info = {
                "chars": len(pdf_text),
                "lines": len(lines),
                "preview": pdf_text[:200].replace("\n", " ") + "…",
                "security_level": security_level,
                "doc_id": doc_id_part,
                "version": version_num,
            }
            new_meta[key] = info
        except Exception as e:  # noqa: BLE001
            new_meta[key] = {"error": str(e)}

    META_JSON_PATH.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"message": "PDF 추출 완료", "pdf_count": len(pdf_paths), "meta_path": str(META_JSON_PATH)}


# -------------------------------------------------
# 2) 전체 임베딩 & 인제스트
# -------------------------------------------------
async def ingest_embeddings(model_key: str | None = "bge"):
    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}

    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    tokenizer, model, device = _load_embedder(model_key)
    emb_dim = int(_embed_text(tokenizer, model, device, "test").shape[0])

    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

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
    import fitz  # type: ignore

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

    tokenizer, model, device = _load_embedder(req.model or "bge")
    emb_dim = int(_embed_text(tokenizer, model, device, "test").shape[0])
    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    try:
        client.delete(COLLECTION_NAME, filter=f"doc_id == '{doc_id}' && version <= {version}")
    except Exception:  # noqa: BLE001
        pass

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

        full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        snippet = chunk_text(full_txt)[int(cidx)]
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


# -------------------------------------------------
# 4) 컬렉션 삭제
# -------------------------------------------------
async def delete_db():
    client = _client()
    cols = client.list_collections()
    for col in cols:
        client.drop_collection(col)
    return {"message": "삭제 완료(Milvus Lite)", "dropped_collections": cols}
