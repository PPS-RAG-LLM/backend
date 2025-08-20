import sqlite3
from utils import get_db, logger, now_kst_string
from errors import DatabaseError


logger = logger(__name__)

def get_chat_history_by_workspace_id(user_id: int, workspace_id: int) -> list[dict]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id FROM workspace_chats WHERE user_id=? AND workspace_id=?", (user_id, workspace_id))
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()

def get_chat_history_by_thread_id(user_id: int, thread_id: int, limit: int) -> list[dict]:
    conn = get_db()
    try:
        conn.row_factory = sqlite3.Row  # Row 객체를 사용하여 딕셔너리처럼 접근 가능
        cur = conn.cursor()
        cur.execute(
            """
            SELECT prompt, response, created_at
            FROM workspace_chats
            WHERE user_id=? AND thread_id=?
            ORDER BY created_at DESC
            LIMIT ?
            """, 
            (user_id, thread_id, limit)
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows] if rows else []
    finally:
        conn.close()


def insert_chat_history(
    user_id: int, category: str, workspace_id: int, prompt: str, response: str, thread_id: int=None
    ) -> int:

    conn= get_db()
    try:
        cur = conn.cursor()
        cur.execute("""INSERT INTO workspace_chats (
            user_id, category, workspace_id, prompt, response, thread_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, 
            (
                user_id, 
                category, 
                workspace_id, 
                prompt, 
                response, 
                thread_id, 
                now_kst_string(), 
                now_kst_string()
            )
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


# def get_chat_history_by_workspace_id(user_id: int, workspace_id: int) -> list[dict]:
