from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
from utils.database import get_db


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


def list_workspace_documents(*, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, doc_id, filename, docpath, workspace_id, metadata, created_at, updated_at
            FROM workspace_documents
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
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
