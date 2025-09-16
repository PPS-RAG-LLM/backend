from __future__ import annotations

import torch, tiktoken, fitz, io, json, uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import UploadFile
from utils import logger, free_torch_memory, now_kst
from config import config
from repository.documents import (
    insert_workspace_document,
    insert_document_vectors,
)
from repository.users.workspace import get_workspace_id_by_slug_for_user
from config import config
from utils.time import now_kst

logger = logger(__name__)

# storage 루트: config의 DB 경로 기준(상대경로 유지)
SAVE_DOC_DIR = config["user_documents"]
DOC_INFO_DIR = Path(SAVE_DOC_DIR["doc_info_dir"])
VEC_CACHE_DIR = Path(SAVE_DOC_DIR["vector_cache_dir"])

# TODO : 임베딩 모델 데이터베이스 조회 후 사용 로직 필요
EMBEDDING_MODEL_DIR = Path(SAVE_DOC_DIR["embedding_model_dir"])


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
) -> tuple[str, Dict[str, Any]]:
    """PDF는 PyMuPDF로 텍스트/메타 추출, 이외는 일반 텍스트 파일로 처리."""
    name_lower = filename.lower()
    meta: Dict[str, Any] = {
        "docAuthor": "Unknown",
        "description": "Unknown",
        "docSource": "a file uploaded by the user.",
    }
    if content_type == "application/pdf" or name_lower.endswith(".pdf"):
        with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
            text_all = "\n\n".join(page.get_text("text").strip() for page in doc)
            pdf_meta = doc.metadata or {}
            meta["docAuthor"] = pdf_meta.get("author") or "Unknown"
            meta["description"] = pdf_meta.get("subject") or "PDF document"
            meta["docSource"] = "pdf file uploaded by the user."
        return text_all, meta
    else:
        # 바이너리를 텍스트로 간주(UTF-8)
        try:
            text_all = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text_all = ""
        meta["description"] = "Text document"
        meta["docSource"] = "a text file uploaded by the user."
        return text_all, meta


def _chunk_text(text: str) -> List[str]:
    # 간단한 토큰 근사: 단어 기준 슬라이딩
    conf = config["vector_defaults"]
    chunk_size = conf["chunk_size"]
    overlap = conf["overlap"]
    words = text.split()
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start += max(chunk_size - overlap, 1)
    return chunks


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    """sentence-transformers로 임베딩 생성. 모델은 config에서 읽고, 없으면 다국어 소형 기본값."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.error(f"sentence_transformers import 실패: {e}")
        raise

    model_path = EMBEDDING_MODEL_DIR
    if not model_path.exists():
        raise FileNotFoundError(f"임베딩 모델 경로를 찾을 수 없음: {model_path}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(str(model_path))
    # model = SentenceTransformer(str(model_path), device="cpu")
    vecs = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=8)
    free_torch_memory()
    return [v.astype(float).tolist() for v in vecs]

    return resp


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
        except Exception as e:
            logger.error(f"upload_documents: failed for {getattr(f, 'filename', None)}: {e}")
            errors.append({"filename": getattr(f, "filename", None), "error": str(e)})
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
    - 파일 저장(원본)
    - documents-info/<filename>-<doc_uuid>.json 작성(pageContent 전체 포함)
    - vector-cache/<doc_uuid>.json 작성(청크별 id, values, metadata.text 포함)
    - DB 기록:
        - workspace_documents: add_to_workspaces가 있을 때만 워크스페이스 매핑
        - document_vectors: doc_id:vector_id를 1:N으로 전부 기록
    """
 
    filename = file.filename or "uploaded"
    content_type = file.content_type or "application/octet-stream"
    file_bytes = await file.read()

    # # 1) 원본 파일 저장
    # saved_path = UPLOAD_DIR / filename
    # with open(saved_path, "wb") as f:
    #     f.write(file_bytes)
    # file_url = f"file://{saved_path.as_posix()}"

    # 2) 텍스트/메타 추출
    page_content, meta = _extract_text_and_meta(file_bytes, filename, content_type)
    word_count = len(page_content.split())
    token_est = _estimate_tokens(page_content)
    now_str = (
        now_kst()
        .strftime("%Y. %m. %d. %p %I:%M:%S")
        .replace("AM", "오전")
        .replace("PM", "오후")
    )

    # 3) 문서 ID 생성 및 documents-info 저장
    doc_id = str(uuid.uuid4())
    doc_info_name = f"{filename}-{doc_id}.json"
    doc_info_path = DOC_INFO_DIR / doc_info_name
    doc_info_path.parent.mkdir(parents=True, exist_ok=True)  # 안전하게 보장

    # TODO : 이미지일 경우 URL 추가
    # doc_info_url = (Path("documents") / "documents-info" / doc_info_name).as_posix()

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
        "pageContent": page_content,
    }
    doc_info_path.write_text(
        json.dumps(doc_info_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 4) 청크 분할 + 임베딩 + vector-cache 저장
    chunks = _chunk_text(page_content)
    if not chunks:
        chunks = [page_content] if page_content else []

    vectors = _embed_chunks(chunks)
    # 청크 메타 템플릿(문서 공통 메타 + chunk text 포함)
    header_meta = (
        "<document_metadata>\n"
        f"sourceDocument: {filename}\n"
        f"published: {now_str}\n"
        "</document_metadata>\n\n"
    )

    vec_items = []
    vector_ids: List[str] = []
    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        vec_id = str(uuid.uuid4())
        vector_ids.append(vec_id)
        vec_items.append(
            {
                "id": vec_id,
                "values": vec,
                "metadata": {
                    "id": doc_id,
                    "url": "",
                    "title": filename,
                    "docAuthor": doc_info_json["docAuthor"],
                    "description": doc_info_json["description"],
                    "docSource": doc_info_json["docSource"],
                    "chunkSource": "",
                    "published": now_str,
                    "wordCount": word_count,
                    "token_count_estimate": token_est,
                    "text": header_meta + chunk_text,
                },
            }
        )

    vec_cache_path = VEC_CACHE_DIR / f"{doc_id}.json"
    vec_cache_path.parent.mkdir(parents=True, exist_ok=True)  # 안전하게 보장
    vec_cache_path.write_text(
        json.dumps(vec_items, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    # 5) DB 기록
    # 5-1) document_vectors: 전체 청크 매핑(항상)

    inserted_workspace = False
    if add_to_workspaces:
        slugs = [s.strip() for s in add_to_workspaces.split(",") if s.strip()]
        for slug in slugs:
            workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
            if not workspace_id:
                logger.info(f"workspace slug not found save only temporarily")
                continue
            if inserted_workspace:
                logger.info(f"doc_id already linked to a workspace, skip additional link : {slug}")
            try:
                insert_workspace_document(
                    doc_id=doc_id,
                    filename=filename,
                    docpath=str(doc_info_path),
                    workspace_id=int(workspace_id),
                    metadata={
                        "chunks": len(vector_ids),
                        "isUserUpload": True,
                    },
                )
                inserted_workspace = True
            except Exception as e:
                logger.error(f"insert_workspace_document failed: {e}")
    if inserted_workspace:
        try:
            insert_document_vectors(doc_id=doc_id, vector_ids=vector_ids)
        except Exception as e:
            logger.error(f"insert_document_vectors failed: {e}")
    # 6) 응답
    resp = {
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
            }
        ],
    }
    return resp



def delete_documents_by_ids(
    doc_ids: List[str], workspace_slug: Optional[str] = None, user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    doc_id 기준으로 documents-info, vector-cache, document_vectors, workspace_documents를 한 번에 삭제.
    """
    from repository.documents import (
        delete_document_vectors_by_doc_ids,
        list_doc_ids_by_workspace,
        delete_workspace_documents_by_doc_ids,
    )

    deleted_files = {"doc_info" : 0, "vector_cache":0}

    # 1) 소유 워크스페이스 확인 (slug → workspace_id)
    if workspace_slug:
        workspace_id = get_workspace_id_by_slug_for_user(user_id, workspace_slug)
        if not workspace_id:
            logger.warning(f"workspace slug not found: {workspace_slug}")
            return {"deleted_doc_ids":[], "deleted_vectors": 0, "deleted_files": deleted_files}
        # 2) 해당 워크스페이스에 속한 doc_id 목록 조회
        try:
            rows = list_doc_ids_by_workspace(int(workspace_id))
            owned_doc_ids = {r.get("doc_id") for r in rows if r and r.get("doc_id")}
        except Exception as e:
            logger.error(f"list_doc_ids_by_workspace failed: {e}")
            return {"deleted_doc_ids":[], "deleted_vectors": 0, "deleted_files": deleted_files}
        # 3) 요청된 doc_ids가 있으면 교집합, 없으면 전부 삭제
        req_doc_ids = [d.strip() for d in (doc_ids or []) if d and isinstance(d, str)]
        req_doc_ids = list(dict.fromkeys(req_doc_ids))
        doc_ids = [d for d in (req_doc_ids or owned_doc_ids) if d in owned_doc_ids]

    else:
        # slug 없으면, 입력 doc_ids만 사용
        doc_ids = [d.strip() for d in (doc_ids or []) if d and isinstance(d, str)]
        doc_ids = list(dict.fromkeys(doc_ids))

    if not doc_ids:
        return {"deleted_doc_ids":[], "deleted_vectors": 0, "deleted_files": deleted_files}

    actually_deleted_docs: List[str] = []
    for did in doc_ids:
        # documents-info/*-{doc_id}.json
        for p in DOC_INFO_DIR.glob(f"*-{did}.json"):
            try: p.unlink()
            except Exception: pass
            deleted_files["doc_info"] += 1

        # vector-cache/{doc_id}.json
        vc =  VEC_CACHE_DIR / f"{did}.json"
        if vc.exists():
            try: vc.unlink()
            except Exception: pass
            deleted_files["vector_cache"] += 1

        actually_deleted_docs.append(did)

    deleted_vectors = 0
    if actually_deleted_docs:
        try:
            # 필요 시 명시 삭제(스키마 CASCADE 설정이 있다면 생략 가능)
            deleted_vectors = delete_document_vectors_by_doc_ids(actually_deleted_docs) or 0
        except Exception as e:
            logger.error(f"delete_document_vectors_by_doc_ids failed: {e}")
        try:
            # workspace_id 검증 포함 삭제
            if workspace_slug:
                delete_workspace_documents_by_doc_ids(actually_deleted_docs, int(workspace_id))
            else:
                # slug 미지정 시 전역 doc_id 삭제(전역 유니크라면 안전)
                delete_workspace_documents_by_doc_ids(actually_deleted_docs, int(-1))  # 필요 시 별도 분기 구현
        except Exception as e:
            logger.error(f"delete_workspace_documents_by_doc_ids failed: {e}")

    return {
            "deleted_doc_ids": actually_deleted_docs,
            "deleted_vectors": int(deleted_vectors),
            "deleted_files": deleted_files,
        }