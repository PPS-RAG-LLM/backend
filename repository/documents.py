from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from utils.database import get_db
from utils import logger

logger = logger(__name__)


def insert_workspace_document(*, doc_id: str, filename: str, docpath: str, workspace_id: int, metadata: Optional[Dict[str, Any]] = None) -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO workspace_documents (doc_id, filename, docpath, workspace_id, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (doc_id, filename, docpath, int(workspace_id), json.dumps(metadata or {}, ensure_ascii=False)),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def insert_document_vectors(*, doc_id: str, vector_ids: List[str])-> int:
    """document_vectors 테이블에 doc_id:vector_id 1:N 매핑 기록"""
    if not vector_ids:
        return 0
    conn = get_db()
    try:
        cur = conn.cursor()
        rows = [(doc_id, v) for v in vector_ids]
        cur.executemany(
            """
            INSERT OR IGNORE INTO document_vectors (doc_id, vector_id)
            VALUES (?, ?)
            """,
            rows,
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def list_doc_ids_by_workspace(workspace_id: int) -> List[Dict[str, Any]]:
    """
    주어진 워크스페이스에 매핑된 문서 doc_id 목록을 반환.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT doc_id
            FROM workspace_documents
            WHERE workspace_id = ?
            """,
            (workspace_id,)
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def list_workspace_documents(workspace_id: int) -> List[Dict[str, Any]]:
    """
    주어진 워크스페이스에 매핑된 문서 메타데이터 목록(doc_id, filename, docpath, metadata)을 반환.
    """
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT doc_id, filename, docpath, metadata
            FROM workspace_documents
            WHERE workspace_id = ?
            ORDER BY id DESC
            """,
            (workspace_id,),
        )
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_workspace_documents_by_filenames(filenames: List[str]) -> int:
    if not filenames:
        return 0
    conn = get_db()
    try:
        cur = conn.cursor()
        placeholders = ",".join(["?"] * len(filenames))
        cur.execute(f"DELETE FROM workspace_documents WHERE filename IN ({placeholders})", filenames)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def delete_document_vectors_by_doc_ids(doc_ids: List[str]) -> int:
    """document_vectors에서 주어진 doc_id들의 벡터만 삭제한다."""
    if not doc_ids:
        return 0
    conn = get_db()
    try:
        cur = conn.cursor()
        q = ",".join(["?"] * len(doc_ids))
        cur.execute(f"DELETE FROM document_vectors WHERE doc_id IN ({q})", doc_ids)
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()

def delete_workspace_documents_by_doc_ids(doc_ids: List[str], workspace_id: int) -> int:
    """
    워크스페이스에 등록된 문서의 경우, 
    workspace_documents에서 주어진 doc_id들의 행을 삭제한다. 
    (document_vectors정보는 자동삭제) 
    """
    if not doc_ids:
        return 0
    conn = get_db()
    try:
        cur = conn.cursor()
        q = ",".join(["?"] * len(doc_ids))
        cur.execute(
            f"DELETE FROM workspace_documents WHERE doc_id IN ({q}) AND workspace_id = ?", 
            [*doc_ids, int(workspace_id)]
        )
        conn.commit()
        return int(cur.rowcount)
    finally:
        conn.close()