from __future__ import annotations

import io
import json
import tiktoken
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from fastapi import UploadFile

from config import config
from errors import DocumentProcessingError
from repository.documents import (
    bulk_upsert_document_metadata,
    delete_document_vectors_by_doc_ids,
    delete_documents,
    delete_workspace_documents_by_doc_ids,
    fetch_document_vectors,
    insert_document_vectors,
    insert_workspace_document,
    list_doc_ids_by_workspace,
    upsert_document,
)
from repository.workspace import (
    get_workspace_id_by_slug_for_user,
    update_workspace_vector_count,
)
from service.vector_db.milvus_store import (
    ensure_collection_and_index,
    get_milvus_client,
    resolve_collection,
)
from storage.db_models import DocumentType
from repository.rag_settings import get_rag_settings_row
from utils import load_embedding_model, logger, make_safe_filename, now_kst

logger = logger(__name__)

# storage 루트: config의 DB 경로 기준(상대경로 유지)
SAVE_DOC_DIR = config["user_documents"]
DOC_INFO_DIR = Path(SAVE_DOC_DIR["doc_info_dir"])
VEC_CACHE_DIR = Path(SAVE_DOC_DIR["vector_cache_dir"])

# TODO : 임베딩 모델 데이터베이스 조회 후 사용 로직 필요
EMBEDDING_MODEL_DIR = Path(SAVE_DOC_DIR["embedding_model_dir"])
TEMP_DOC_CFG = config.get("temp_documents", {}) or {}
TEMP_TTL_HOURS = int(TEMP_DOC_CFG.get("ttl_hours", 12))

def _get_embedding_key() -> str:
       rag = get_rag_settings_row()
       return rag.get("embedding_key")

def _estimate_tokens(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = None
    if enc is None:
        # 대략치: 영문 1.3, 한글 2.0 근사 → 1.6로 보수치
        return int(len(text) / 1.6)
    return len(enc.encode(text))


def _extract_text_and_meta(
    file_bytes: bytes, filename: str, content_type: str
) -> tuple[List[str], Dict[str, Any]]:
    """PDF는 페이지별 텍스트를 반환하고, 기타 문서는 단일 페이지 리스트로 감싼다."""
    name_lower = filename.lower()
    meta: Dict[str, Any] = {
        "docAuthor": "Unknown",
        "description": "Unknown",
        "docSource": "a file uploaded by the user.",
    }
    if content_type == "application/pdf" or name_lower.endswith(".pdf"):
        with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
            # 각 페이지 텍스트 추출
            page_texts = []
            for page in doc:
                text = page.get_text("text").strip()
                page_texts.append(text)
            
            # 텍스트가 거의 없으면 경고 로그
            text_all = "\n\n".join(page_texts)
            if len(text_all.strip()) < 10:
                logger.warning(
                    f"PDF '{filename}': 추출된 텍스트가 거의 없습니다 ({len(text_all)} chars). "
                    "이미지 기반 PDF일 가능성이 있습니다. OCR이 필요할 수 있습니다."
                )
            
            pdf_meta = doc.metadata or {}
            meta["docAuthor"] = pdf_meta.get("author") or "Unknown"
            meta["description"] = pdf_meta.get("subject") or "PDF document"
            meta["docSource"] = "pdf file uploaded by the user."
        return page_texts or [""], meta
    else:
        # 바이너리를 텍스트로 간주(UTF-8)
        try:
            text_all = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text_all = ""
        meta["description"] = "Text document"
        meta["docSource"] = "a text file uploaded by the user."
        return [text_all], meta


def _build_chunk_records(page_texts: List[str]) -> List[Dict[str, Any]]:
    """페이지 텍스트 리스트를 페이지/청크 단위 레코드로 변환."""
    rag = get_rag_settings_row()
    chunk_size = int(rag.get("chunk_size", 250))
    overlap = int(rag.get("overlap", 50))
    step = max(1, chunk_size - overlap)
    global_idx = 0
    records: List[Dict[str, Any]] = []

    for page_idx, page_text in enumerate(page_texts, start=1):
        words = page_text.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                records.append(
                    {
                        "page": page_idx,
                        "chunk_index": global_idx,
                        "text": chunk,
                    }
                )
                global_idx += 1
            if end == len(words):
                break
            start += step

    if not records:
        for page_idx, page_text in enumerate(page_texts, start=1):
            trimmed = page_text.strip()
            if not trimmed:
                continue
            records.append(
                {
                    "page": page_idx,
                    "chunk_index": global_idx,
                    "text": trimmed,
                }
            )
            global_idx += 1
    return records


async def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    """sentence-transformers로 임베딩 생성. 모델은 config에서 읽고, 없으면 다국어 소형 기본값."""
    import asyncio
    try:
        model_path = EMBEDDING_MODEL_DIR
        if not model_path.exists():
            raise FileNotFoundError(f"임베딩 모델 경로를 찾을 수 없음: {model_path}")

        # 동기 블로킹 작업을 별도 쓰레드에서 실행하여 이벤트 루프 차단 방지
        def _sync_embed():
            model = load_embedding_model()
            vecs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=8)
            return [v.astype(float).tolist() for v in vecs]
        
        # asyncio.to_thread로 동기 작업을 비동기로 처리
        return await asyncio.to_thread(_sync_embed)
    except FileNotFoundError as e:
        logger.error(f"임베딩 모델 경로를 찾을 수 없음: {e}")
        raise DocumentProcessingError("embedding_model", str(e)) from e
    except Exception as exc:
        logger.exception("임베딩 계산 실패")
        raise DocumentProcessingError("embedding", str(exc)) from exc



# 아래는 여러 파일 업로드 지원 함수
async def upload_documents(
    *,
    user_id: int,
    files: List[UploadFile],
    add_to_workspaces: Optional[str],
) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for f in files:
        try:
            res = await upload_document(
                user_id=user_id,
                file=f,
                add_to_workspaces=add_to_workspaces,
            )
            if res and res.get("documents"):
                documents.extend(res["documents"])
        except DocumentProcessingError as exc:
            logger.error(f"upload_documents: stage failure for {f.filename}: {exc.message}")
            errors.append({"filename": getattr(f, "filename", None), "error": exc.message})
        except Exception as exc:
            logger.exception("upload_documents: unexpected failure")
            errors.append({"filename": getattr(f, "filename", None), "error": str(exc)})
    return {
        "success": len(errors) == 0,
        "error": errors or None,
        "documents": documents,
    }

async def upload_document(
    *,
    user_id: int,
    file: UploadFile,
    add_to_workspaces: Optional[str],
) -> Dict[str, Any]:
    """
    사용자 업로드 문서를 RDB + Milvus 에 저장한다.
    """

    filename = file.filename or "uploaded"
    content_type = file.content_type or "application/octet-stream"
    file_bytes = await file.read()

    try:
        page_texts, meta = _extract_text_and_meta(file_bytes, filename, content_type)
    except Exception as exc:
        logger.exception("텍스트 추출 실패")
        raise DocumentProcessingError("extract_text", str(exc)) from exc

    combined_text = "\n\n".join(page_texts).strip()
    if not combined_text:
        raise DocumentProcessingError("extract_text", "추출된 텍스트가 없습니다.")

    word_count = len(combined_text.split())
    token_est = _estimate_tokens(combined_text)
    now_str = (
        now_kst()
        .strftime("%Y. %m. %d. %p %I:%M:%S")
        .replace("AM", "오전")
        .replace("PM", "오후")
    )

    doc_id = str(uuid.uuid4())
    doc_info_name = make_safe_filename(filename, doc_id, "json")
    doc_info_path = DOC_INFO_DIR / doc_info_name
    doc_info_path.parent.mkdir(parents=True, exist_ok=True)
    doc_info_json = {
        "id": doc_id,
        "url": "",
        "title": filename,
        "docAuthor": meta.get("docAuthor") or "Unknown",
        "description": meta.get("description") or "Unknown",
        "docSource": meta.get("docSource") or "Unknown",
        "chunkSource": filename,
        "published": now_str,
        "wordCount": word_count,
        "token_count_estimate": token_est,
        "pageContent": combined_text,
    }
    doc_info_path.write_text(
        json.dumps(doc_info_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        chunk_records = _build_chunk_records(page_texts)
    except Exception as exc:
        logger.exception("청크 분할 실패")
        raise DocumentProcessingError("chunking", str(exc)) from exc
    if not chunk_records:
        raise DocumentProcessingError("chunking", "생성된 청크가 없습니다.")

    try:
        vectors = await _embed_chunks([rec["text"] for rec in chunk_records])
    except DocumentProcessingError:
        raise
    except Exception as exc:
        logger.exception("임베딩 단계 실패")
        raise DocumentProcessingError("embedding", str(exc)) from exc

    target_workspace_id: Optional[int] = None
    if add_to_workspaces:
        slugs = [s.strip() for s in add_to_workspaces.split(",") if s.strip()]
        for slug in slugs:
            workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
            if not workspace_id:
                logger.info("workspace slug not found: %s", slug)
                continue
            target_workspace_id = int(workspace_id)
            break

    doc_type = DocumentType.WORKSPACE if target_workspace_id is not None else DocumentType.TEMP
    doc_payload: Dict[str, Any] = {
        "docAuthor": doc_info_json["docAuthor"],
        "doc_info_path": str(doc_info_path),
        "uploaded_at": now_str,
    }
    if doc_type is DocumentType.TEMP:
        expires_at = (
            datetime.now(timezone.utc) + timedelta(hours=TEMP_TTL_HOURS)
        ).isoformat()
        doc_payload["expires_at"] = expires_at

    try:
        upsert_document(
            doc_id=doc_id,
            doc_type=doc_type.value,
            filename=filename,
            payload=doc_payload,
            workspace_id=target_workspace_id,
        )
        bulk_upsert_document_metadata(
            doc_id=doc_id,
            records=[
                {
                    "page": rec["page"],
                    "chunk_index": rec["chunk_index"],
                    "text": rec["text"],
                    "payload": {
                        "title": filename,
                        "published": now_str,
                    },
                }
                for rec in chunk_records
            ],
        )
    except Exception as exc:
        logger.exception("문서 메타데이터 저장 실패")
        raise DocumentProcessingError("database", str(exc)) from exc

    collection_name = resolve_collection(doc_type.value)
    client = get_milvus_client()
    emb_dim = len(vectors[0]) if vectors else 0
    ensure_collection_and_index(
        client,
        emb_dim=emb_dim,
        metric="IP",
        collection_name=collection_name,
    )

    milvus_rows = []
    for rec, vec in zip(chunk_records, vectors):
        milvus_rows.append(
            {
                "embedding": vec,
                "path": f"documents-info/{doc_info_name}",
                "chunk_idx": int(rec["chunk_index"]),
                "task_type": "qna",
                "security_level": 0,
                "doc_id": doc_id,
                "version": 1,
                "page": int(rec["page"]),
                "workspace_id": int(workspace_id or 0),
                "text": rec["text"],
            }
        )

    try:
        insert_result = client.insert(collection_name=collection_name, data=milvus_rows)
        try:
            client.flush(collection_name)
        except Exception:
            pass
    except Exception as exc:
        logger.exception("Milvus insert 실패")
        raise DocumentProcessingError("vector_store", str(exc)) from exc

    primary_keys_raw: List[Any] = []
    if isinstance(insert_result, dict):
        ids = insert_result.get("ids") or insert_result.get("primary_keys")
        if ids:
            primary_keys_raw = list(ids)
    else:
        primary_keys_raw = list(
            getattr(insert_result, "primary_keys", [])
            or getattr(insert_result, "ids", [])
        )

    primary_keys = [str(pk) for pk in primary_keys_raw if pk is not None]
    vector_records = []
    for idx, rec in enumerate(chunk_records):
        vector_id = primary_keys[idx] if idx < len(primary_keys) else None
        if vector_id is None:
            logger.warning(
                "[upload_document] primary key 누락: doc_id=%s chunk_idx=%s",
                doc_id,
                rec["chunk_index"],
            )
            continue
        vector_records.append(
            {
                "vector_id": vector_id,
                "page": rec["page"],
                "chunk_index": rec["chunk_index"],
            }
        )
    try:
        if vector_records:
            insert_document_vectors(
                doc_id=doc_id,
                collection=collection_name,
                embedding_version=_get_embedding_key(),
                vectors=vector_records,
            )
    except Exception as exc:
        logger.exception("document_vectors 기록 실패")
        raise DocumentProcessingError("database", str(exc)) from exc

    if target_workspace_id is not None:
        try:
            insert_workspace_document(
                doc_id=doc_id,
                filename=filename,
                docpath=str(doc_info_path),
                workspace_id=target_workspace_id,
                metadata={
                    "chunks": len(chunk_records),
                    "isUserUpload": True,
                },
            )
        except Exception as exc:
            logger.error("insert_workspace_document 실패: %s", exc)
        else:
            update_workspace_vector_count(target_workspace_id)

    return {
        "success": True,
        "error": None,
        "documents": [
            {
                "location": f"documents-info/{doc_info_name}",
                "name": doc_info_name,
                "url": "",
                "title": filename,
                "docAuthor": doc_info_json["docAuthor"],
                "description": doc_info_json["description"],
                "docSource": doc_info_json["docSource"],
                "chunkSource": filename,
                "published": now_str,
                "wordCount": word_count,
                "token_count_estimate": token_est,
                "docType": doc_type.value,
            }
        ],
    }



def delete_documents_by_ids(
    doc_ids: List[str],
    workspace_slug: Optional[str] = None,
    user_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    doc_id 기준으로 문서 메타/벡터/Milvus 레코드를 삭제한다.
    """

    deleted_files = {"doc_info": 0, "vector_cache": 0}
    workspace_id: Optional[int] = None

    cleaned_ids = [d.strip() for d in (doc_ids or []) if isinstance(d, str) and d.strip()]
    doc_ids = list(dict.fromkeys(cleaned_ids))

    if workspace_slug:
        workspace_id = get_workspace_id_by_slug_for_user(user_id, workspace_slug)
        if not workspace_id:
            logger.warning("workspace slug not found: %s", workspace_slug)
            return {"deleted_doc_ids": [], "deleted_vectors": 0, "deleted_files": deleted_files}
        try:
            rows = list_doc_ids_by_workspace(int(workspace_id))
        except Exception as exc:
            logger.error("list_doc_ids_by_workspace 실패: %s", exc)
            return {"deleted_doc_ids": [], "deleted_vectors": 0, "deleted_files": deleted_files}
        owned_doc_ids = {r.get("doc_id") for r in rows if r and r.get("doc_id")}
        targets = doc_ids or list(owned_doc_ids)
        doc_ids = [d for d in targets if d in owned_doc_ids]

    if not doc_ids:
        return {"deleted_doc_ids": [], "deleted_vectors": 0, "deleted_files": deleted_files}

    deleted_files = delete_document_files(doc_ids)

    vectors = fetch_document_vectors(doc_ids)
    deleted_vectors = 0
    if vectors:
        client = get_milvus_client()
        grouped: Dict[str, List[str]] = {}
        for row in vectors:
            grouped.setdefault(row["collection"], []).append(row["vector_id"])
        for collection, vec_ids in grouped.items():
            numeric_ids: List[int] = []
            for vid in vec_ids:
                try:
                    numeric_ids.append(int(vid))
                except (TypeError, ValueError):
                    logger.warning("invalid vector_id for deletion: %s", vid)
            if not numeric_ids:
                continue
            expr = f"pk in [{', '.join(str(x) for x in numeric_ids)}]"
            try:
                client.delete(collection_name=collection, filter=expr)
                deleted_vectors += len(numeric_ids)
            except Exception as exc:
                logger.error("Milvus delete 실패(collection=%s): %s", collection, exc)

    delete_document_vectors_by_doc_ids(doc_ids)
    delete_workspace_documents_by_doc_ids(doc_ids, workspace_id)
    delete_documents(doc_ids)

    if workspace_id is not None:
        try:
            update_workspace_vector_count(int(workspace_id))
        except Exception as exc:
            logger.error("update_workspace_vector_count 실패: %s", exc)

    return {
        "deleted_doc_ids": doc_ids,
        "deleted_vectors": int(deleted_vectors),
        "deleted_files": deleted_files,
    }

def delete_document_files(doc_ids: List[str]) -> Dict[str, int]:
    """
    doc_id 리스트에 해당하는 파일들(doc_info, vector_cache)을 삭제.
    Returns: {"doc_info": count, "vector_cache": count}
    """
    deleted_files = {"doc_info": 0, "vector_cache": 0}
    
    for did in doc_ids:
        # documents-info/*-{doc_id}.json
        for p in DOC_INFO_DIR.glob(f"*-{did}.json"):
            try:
                p.unlink()
                deleted_files["doc_info"] += 1
            except Exception as e:
                logger.warning(f"Failed to delete doc_info file {p}: {e}")

        # vector-cache/{doc_id}.json
        vc = VEC_CACHE_DIR / f"{did}.json"
        if vc.exists():
            try:
                vc.unlink()
                deleted_files["vector_cache"] += 1
            except Exception as e:
                logger.warning(f"Failed to delete vector_cache file {vc}: {e}")
    
    return deleted_files