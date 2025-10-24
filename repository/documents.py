from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from repository.workspace import update_workspace_vector_count
from utils import logger
from utils.database import get_session
from storage.db_models import WorkspaceDocument, DocumentVector
from sqlalchemy import select, delete, desc
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

logger = logger(__name__)


def insert_workspace_document(*, doc_id: str, filename: str, docpath: str, workspace_id: int, metadata: Optional[Dict[str, Any]] = None) -> int:
    """workspace_documents에 1건 추가 후 PK(id) 반환"""
    with get_session() as session:
        obj = WorkspaceDocument(
            doc_id=doc_id,
            filename=filename,
            docpath=docpath,
            workspace_id=int(workspace_id),
            metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        )
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return int(obj.id)


def insert_document_vectors(*, doc_id: str, vector_ids: List[str]) -> int:
    """document_vectors에 doc_id:vector_id 1:N 매핑 기록 (중복 무시)"""
    if not vector_ids:
        return 0
    with get_session() as session:
        rows = [{"doc_id": doc_id, "vector_id": v} for v in vector_ids]
        stmt = sqlite_insert(DocumentVector).values(rows).on_conflict_do_nothing(index_elements=["doc_id", "vector_id"])
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


def list_doc_ids_by_workspace(workspace_id: int) -> List[Dict[str, Any]]:
    """주어진 워크스페이스에 매핑된 문서 doc_id 목록을 반환."""
    with get_session() as session:
        stmt = select(WorkspaceDocument.doc_id).where(WorkspaceDocument.workspace_id == workspace_id)
        rows = session.execute(stmt).all()
        return [{"doc_id": r[0]} for r in rows]


def list_workspace_documents(workspace_id: int) -> List[Dict[str, Any]]:
    """주어진 워크스페이스에 매핑된 문서 메타데이터 목록(doc_id, filename, docpath, metadata)을 반환."""
    with get_session() as session:
        stmt = (
            select(WorkspaceDocument)
            .where(WorkspaceDocument.workspace_id == workspace_id)
            .order_by(desc(WorkspaceDocument.id))
        )
        rows = session.execute(stmt).scalars().all()
        return [
            {
                "doc_id": r.doc_id,
                "filename": r.filename,
                "docpath": r.docpath,
                "metadata": json.loads(r.metadata_json or "{}"),
            }
            for r in rows
        ]


def delete_workspace_documents_by_filenames(filenames: List[str]) -> int:
    if not filenames:
        return 0
    with get_session() as session:
        stmt = delete(WorkspaceDocument).where(WorkspaceDocument.filename.in_(filenames))
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

def delete_workspace_documents_by_doc_ids(doc_ids: List[str], workspace_id: int) -> int:
    """
    워크스페이스에 등록된 문서의 경우, 
    workspace_documents에서 주어진 doc_id들의 행을 삭제한다. 
    (document_vectors정보는 FK CASCADE로 자동삭제) 
    """
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = delete(WorkspaceDocument).where(
            WorkspaceDocument.doc_id.in_(doc_ids),
            WorkspaceDocument.workspace_id == int(workspace_id),
        )
        result = session.execute(stmt)
        session.commit()
        return int(result.rowcount or 0)


