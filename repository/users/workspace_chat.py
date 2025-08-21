import sqlite3
from utils import get_db, logger, now_kst_string
from errors import DatabaseError


logger = logger(__name__)

def get_chat_history_by_workspace_id(user_id: int, workspace_id: int, limit: int | None = None) -> list[dict]:
    conn = get_db()
    try:
        cur = conn.cursor()
        base_sql = """
            SELECT id, category, thread_id, prompt, response, created_at
            FROM workspace_chats
            WHERE user_id=? AND workspace_id=?
            ORDER BY created_at DESC
        """
        if limit and limit > 0:
            cur.execute(base_sql + " LIMIT ?", (user_id, workspace_id, limit))
        else:
            cur.execute(base_sql, (user_id, workspace_id))
        rows = cur.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        conn.close()

def get_chat_history_by_thread_id(user_id: int, thread_id: int, limit: int) -> list[dict]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT prompt, response, created_at
            FROM workspace_chats
            WHERE user_id=? AND thread_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, thread_id, limit),
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        conn.close()

def insert_chat_history(
    user_id: int, category: str, workspace_id: int, prompt: str, response: str, thread_id: int | None = None
) -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO workspace_chats (
                user_id, category, workspace_id, prompt, response, thread_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id, category, workspace_id, prompt, response, thread_id,
                now_kst_string(), now_kst_string(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError as exc:
        logger.error(f"insert_chat_history failed: {exc}")
        raise DatabaseError(str(exc))
    finally:
        conn.close()


