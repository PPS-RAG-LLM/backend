from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import delete, desc, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from utils import logger, now_kst, now_kst_string
from utils.database import get_session
from sqlalchemy import func
from storage.db_models import (
    Document,
    DocumentMetadata,
    DocumentType,
    DocumentVector,
)

logger = logger(__name__)


def _serialize_document(row: Document) -> Dict[str, Any]:
    return {
        "id": row.id,
        "doc_id": row.doc_id,
        "doc_type": row.doc_type,
        "filename": row.filename,
        "workspace_id": row.workspace_id,
        "source_path": row.source_path,
        "payload": row.payload or {},
        "security_level": row.security_level,
        "updated_at": row.updated_at,
        "created_at": row.created_at,
    }


def upsert_document(
    *,
    doc_id: str,
    doc_type: str,
    filename: str,
    payload: Optional[Dict[str, Any]] = None,
    workspace_id: Optional[int] = None,
    security_level: Optional[int] = None,
    source_path: Optional[str] = None,
) -> None:
    """documents 테이블에 공통 메타데이터를 upsert."""
    payload = payload or {}
    with get_session() as session:
        insert_stmt = sqlite_insert(Document).values(
            doc_id=doc_id,
            doc_type=doc_type,
            workspace_id=workspace_id,
            security_level=security_level,
            filename=filename,
            source_path=source_path,
            payload=payload,
        )
        update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[Document.doc_id],
            set_={
                "doc_type": insert_stmt.excluded.doc_type,
                "workspace_id": insert_stmt.excluded.workspace_id,
                "security_level": insert_stmt.excluded.security_level,
                "filename": insert_stmt.excluded.filename,
                "source_path": insert_stmt.excluded.source_path,
                "payload": insert_stmt.excluded.payload,
            },
        )
        session.execute(update_stmt)
        session.commit()


def bulk_upsert_document_metadata(
    *,
    doc_id: str,
    records: List[Dict[str, Any]],
) -> int:
    """document_metadata에 페이지/청크 단위 메타데이터를 upsert."""
    if not records:
        return 0
    rows = []
    for record in records:
        rows.append(
            {
                "doc_id": doc_id,
                "page": int(record.get("page", 0)),
                "chunk_index": int(record.get("chunk_index", 0)),
                "text": record.get("text") or "",
                "payload": record.get("payload") or {},
            }
        )
    with get_session() as session:
        insert_stmt = sqlite_insert(DocumentMetadata).values(rows)
        update_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[
                DocumentMetadata.doc_id,
                DocumentMetadata.page,
                DocumentMetadata.chunk_index,
            ],
            set_={
                "text": insert_stmt.excluded.text,
                "payload": insert_stmt.excluded.payload,
            },
        )
        result = session.execute(update_stmt)
        session.commit()
        return int(result.rowcount or 0)


def insert_workspace_document(
    *,
    doc_id: str,
    filename: str,
    docpath: str,
    workspace_id: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """
    documents 테이블에 워크스페이스 전용 메타데이터를 병합한다.
    기존 row가 없으면 ValueError를 발생시킨다(업로드 흐름에서 선행 upsert 필요).
    """
    metadata = metadata or {}
    with get_session() as session:
        stmt = select(Document).where(Document.doc_id == doc_id).limit(1)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is None:
            raise ValueError(f"Document not found for doc_id={doc_id}")

        payload = dict(existing.payload or {})
        payload.setdefault("doc_info_path", docpath)
        payload["workspace_metadata"] = metadata

        existing.filename = filename
        existing.workspace_id = int(workspace_id)
        existing.doc_type = DocumentType.WORKSPACE.value
        existing.payload = payload
        existing.updated_at = now_kst()

        session.commit()
        return int(existing.id)


def insert_document_vectors(
    *,
    doc_id: str,
    collection: str,
    embedding_version: str,
    vectors: List[Dict[str, Any]],
) -> int:
    """document_vectors에 doc_id와 Milvus vector pk 매핑을 기록."""
    if not vectors:
        return 0
    rows = []
    for item in vectors:
        rows.append(
            {
                "doc_id": doc_id,
                "vector_id": str(item["vector_id"]),
                "task_type": str(item.get("task_type", "")),
                "collection": collection,
                "embedding_version": embedding_version,
                "page": item.get("page"),
                "chunk_index": item.get("chunk_index"),
            }
        )
    with get_session() as session:
        insert_stmt = sqlite_insert(DocumentVector).values(rows)
        insert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[
                DocumentVector.doc_id,
                DocumentVector.vector_id,
            ],
            set_={
                "collection": insert_stmt.excluded.collection,
                "task_type": insert_stmt.excluded.task_type,
                "embedding_version": insert_stmt.excluded.embedding_version,
                "page": insert_stmt.excluded.page,
                "chunk_index": insert_stmt.excluded.chunk_index,
            },
        )
        result = session.execute(insert_stmt)
        session.commit()
        return int(result.rowcount or 0)


def delete_document_vectors(
    doc_id: str,
    task_type: Optional[str] = None,
) -> int:
    """document_vectors에서 해당 doc_id(및 task_type)의 벡터를 삭제."""
    with get_session() as session:
        stmt = delete(DocumentVector).where(DocumentVector.doc_id == doc_id)
        if task_type:
            stmt = stmt.where(DocumentVector.task_type == task_type)
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def document_has_vectors(doc_id: str) -> bool:
    """doc_id에 남아 있는 벡터가 있는지 확인."""
    with get_session() as session:
        stmt = select(func.count(DocumentVector.vector_id)).where(DocumentVector.doc_id == doc_id)
        count = session.execute(stmt).scalar() or 0
        return count > 0


def list_doc_ids_by_workspace(workspace_id: int) -> List[Dict[str, Any]]:
    """주어진 워크스페이스에 매핑된 문서 doc_id 목록을 반환."""
    with get_session() as session:
        stmt = (
            select(Document.doc_id)
            .where(
                Document.workspace_id == workspace_id,
                Document.doc_type == DocumentType.WORKSPACE.value,
            )
        )
        rows = session.execute(stmt).all()
        return [{"doc_id": r[0]} for r in rows]


def list_workspace_documents(workspace_id: int) -> List[Dict[str, Any]]:
    """주어진 워크스페이스에 매핑된 문서 메타데이터 목록(doc_id, filename, docpath, metadata)을 반환."""
    with get_session() as session:
        stmt = (
            select(Document)
            .where(
                Document.workspace_id == workspace_id,
                Document.doc_type == DocumentType.WORKSPACE.value,
            )
            .order_by(desc(Document.id))
        )
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "doc_id": r.doc_id,
                "filename": r.filename,
                "docpath": (r.payload or {}).get("doc_info_path"),
                "metadata": (r.payload or {}).get("workspace_metadata") or {},
            }
            for r in rows
        ]


def delete_workspace_documents_by_filenames(filenames: List[str]) -> int:
    if not filenames:
        return 0
    with get_session() as session:
        stmt = delete(Document).where(
            Document.doc_type == DocumentType.WORKSPACE.value,
            Document.filename.in_(filenames),
        )
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def delete_document_vectors_by_doc_ids(doc_ids: List[str]) -> int:
    """document_vectors에서 주어진 doc_id들의 벡터만 삭제한다."""
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = delete(DocumentVector).where(DocumentVector.doc_id.in_(doc_ids))
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def fetch_document_vectors(doc_ids: List[str]) -> List[Dict[str, Any]]:
    """doc_id에 해당하는 vector 메타데이터 목록을 반환."""
    if not doc_ids:
        return []
    with get_session() as session:
        stmt = (
            select(
                DocumentVector.doc_id,
                DocumentVector.vector_id,
                DocumentVector.collection,
            )
            .where(DocumentVector.doc_id.in_(doc_ids))
            .order_by(DocumentVector.doc_id)
        )
        rows = session.execute(stmt).all()
        return [
            {
                "doc_id": doc_id,
                "vector_id": vector_id,
                "collection": collection,
            }
            for doc_id, vector_id, collection in rows
        ]


def get_documents_by_ids(doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """documents 테이블에서 doc_id 매핑을 조회."""
    if not doc_ids:
        return {}
    with get_session() as session:
        stmt = select(Document).where(Document.doc_id.in_(doc_ids))
        rows = session.execute(stmt).scalars().all()
        return {
            row.doc_id: {
                "doc_id": row.doc_id,
                "doc_type": row.doc_type,
                "filename": row.filename,
                "workspace_id": row.workspace_id,
                "payload": row.payload or {},
            }
            for row in rows
        }

def delete_workspace_documents_by_doc_ids(
    doc_ids: List[str], workspace_id: Optional[int] = None
) -> int:
    """
    워크스페이스에 등록된 문서의 경우, 
    workspace_documents에서 주어진 doc_id들의 행을 삭제한다. 
    (document_vectors정보는 FK CASCADE로 자동삭제) 
    """
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = delete(Document).where(
            Document.doc_type == DocumentType.WORKSPACE.value,
            Document.doc_id.in_(doc_ids),
        )
        if workspace_id is not None:
            stmt = stmt.where(Document.workspace_id == int(workspace_id))
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def delete_documents(doc_ids: List[str]) -> int:
    """documents 테이블에서 doc_id 목록을 삭제."""
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = delete(Document).where(Document.doc_id.in_(doc_ids))
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def list_documents_by_type(
    doc_type: str,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """지정 doc_type의 문서 메타데이터 목록을 반환한다."""
    with get_session() as session:
        stmt = (
            select(Document)
            .where(Document.doc_type == doc_type)
            .order_by(desc(Document.updated_at))
            .offset(max(0, int(offset)))
        )
        if limit is not None:
            stmt = stmt.limit(max(0, int(limit)))
        rows = session.execute(stmt).scalars().all()
        return [_serialize_document(r) for r in rows]


def delete_documents_by_type_and_ids(doc_type: str, doc_ids: List[str]) -> int:
    """특정 doc_type에 속한 문서 중 doc_id 목록을 삭제한다."""
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = delete(Document).where(
            Document.doc_type == doc_type,
            Document.doc_id.in_(doc_ids),
        )
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def delete_documents_not_in_doc_ids(doc_type: str, doc_ids: List[str]) -> int:
    """
    doc_type에 해당하는 문서들 중 전달된 doc_id 목록에 포함되지 않는 항목을 삭제한다.
    doc_ids가 비어 있으면 해당 doc_type 전체를 삭제한다.
    """
    with get_session() as session:
        stmt = delete(Document).where(Document.doc_type == doc_type)
        if doc_ids:
            stmt = stmt.where(~Document.doc_id.in_(doc_ids))
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def get_document_by_source_path(
    doc_type: str,
    source_path: str,
) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        stmt = (
            select(Document)
            .where(
                Document.doc_type == doc_type,
                Document.source_path == source_path,
            )
            .limit(1)
        )
        row = session.execute(stmt).scalar_one_or_none()
        if not row:
            return None
        return _serialize_document(row)


def get_list_indexed_files(collection_name: str, offset: int = 0, limit: int = 1000, task_type: Optional[str] = None):
    with get_session() as session:
        stmt = (
            select(
                DocumentVector.doc_id,
                DocumentVector.task_type,
                func.count(DocumentVector.vector_id).label("chunk_count"),
            )
            .where(DocumentVector.collection == collection_name)
            .group_by(DocumentVector.doc_id, DocumentVector.task_type)
            .offset(max(0, offset))
            .limit(limit)
        )
        if task_type:
            stmt = stmt.where(DocumentVector.task_type == task_type)
        rows = session.execute(stmt).all()
        return rows


def purge_documents_by_collection(doc_types: Sequence[str]) -> Dict[str, int]:
    """
    전달된 doc_type 목록과 연결된 문서·메타데이터·벡터를 모두 제거한다.
    doc_types가 비어 있으면 아무 작업도 수행하지 않는다.
    """
    if isinstance(doc_types, str):
        doc_types = [doc_types]
    doc_types = [dt for dt in doc_types if dt]
    stats = {"vectors": 0, "metadata": 0, "documents": 0}

    if not doc_types:
        return stats

    with get_session() as session:
        admin_ids_stmt = select(Document.doc_id).where(Document.doc_type.in_(doc_types))

        del_vectors = delete(DocumentVector).where(DocumentVector.doc_id.in_(admin_ids_stmt))
        res_vec = session.execute(del_vectors)
        stats["vectors"] = int(res_vec.rowcount or 0)

        del_metadata = delete(DocumentMetadata).where(DocumentMetadata.doc_id.in_(admin_ids_stmt))
        res_meta = session.execute(del_metadata)
        stats["metadata"] = int(res_meta.rowcount or 0)

        del_docs = delete(Document).where(Document.doc_type.in_(doc_types))
        res_docs = session.execute(del_docs)
        stats["documents"] = int(res_docs.rowcount or 0)

        session.commit()

    return stats



def fetch_metadata_by_vector_ids(vector_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    vector_id 목록을 DocumentMetadata/DocumentVector 조합으로 조회해
    doc_id·chunk_index·text 등을 돌려준다.
    """
    if not vector_ids:
        return {}
    with get_session() as session:
        stmt = (
            select(
                DocumentVector.vector_id,
                DocumentVector.doc_id,
                DocumentVector.chunk_index,
                DocumentMetadata.text,
                DocumentMetadata.payload,
            )
            .join(
                DocumentMetadata,
                (DocumentMetadata.doc_id == DocumentVector.doc_id)
                & (DocumentMetadata.chunk_index == DocumentVector.chunk_index),
            )
            .where(DocumentVector.vector_id.in_(vector_ids))
        )
        rows = session.execute(stmt).all()
        return {
            str(vector_id): {
                "doc_id": doc_id,
                "chunk_index": chunk_index,
                "text": text or "",
                "payload": payload or {},
            }
            for vector_id, doc_id, chunk_index, text, payload in rows
        }


def fetch_document_metadata_by_doc_ids(doc_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    주어진 doc_id 목록에 대한 페이지/청크 메타데이터를 모두 조회해
    {doc_id: [{chunk_index, page, text, payload}, ...]} 형태로 반환한다.
    """
    if not doc_ids:
        return {}

    with get_session() as session:
        stmt = (
            select(
                DocumentMetadata.doc_id,
                DocumentMetadata.chunk_index,
                DocumentMetadata.page,
                DocumentMetadata.text,
                DocumentMetadata.payload,
            )
            .where(DocumentMetadata.doc_id.in_(doc_ids))
            .order_by(DocumentMetadata.doc_id, DocumentMetadata.chunk_index)
        )
        rows = session.execute(stmt).all()

    result: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id, chunk_idx, page, text, payload in rows:
        result.setdefault(doc_id, []).append(
            {
                "chunk_index": int(chunk_idx),
                "page": int(page or 0),
                "text": text or "",
                "payload": payload or {},
            }
        )
    return result