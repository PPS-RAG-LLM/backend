from utils import get_db, logger
import sqlite3
from errors import DatabaseError
from utils import generate_unique_slug, now_kst_string
from repository.users.workspace import get_workspace_id_by_name

logger = logger(__name__)

### 
def create_default_thread(user_id: int, name: str, thread_slug: str, workspace_id: int=None) -> int:
    """워크스페이스의 기본 스레드를 생성합니다."""
    conn = get_db()
    try:
        cur = conn.cursor()
        if not workspace_id:
            workspace_id = get_workspace_id_by_name(user_id, name)
        cur.execute(
            """
            INSERT INTO workspace_threads (user_id, name, slug, workspace_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, name, thread_slug, workspace_id, now_kst_string(), now_kst_string()),
        )
        thread_id = cur.lastrowid
        conn.commit()
        logger.info(f"Default thread created: id={thread_id}, workspace_id={workspace_id}")
        return thread_id
    except sqlite3.IntegrityError as exc:
        logger.error(f"thread creation failed: {exc}")
        raise DatabaseError(f"thread creation failed: {exc}") from exc
    finally:
        conn.close()

def get_thread_by_id(thread_id: int) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, slug, user_id, workspace_id FROM workspace_threads WHERE id=?
            """,
            (thread_id,),
        ) 
        row = cur.fetchone()
        if row:
            logger.debug(f"Thread fetched: id={thread_id}")
        else:
            logger.warning(f"Thread not found: id={thread_id}")
        return dict(row) if row else None
    finally:
        conn.close()

def get_thread_by_workspace_id(workspace_id: int) -> list[dict]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name, slug, user_id, workspace_id, created_at, updated_at FROM workspace_threads WHERE workspace_id=?
            """,
            (workspace_id,),
        )   
        rows = cur.fetchall()
        if rows:
            logger.debug(f"Threads fetched: workspace_id={workspace_id}")
        else:
            logger.warning(f"No threads found for workspace_id={workspace_id}")
        return [dict(row) for row in rows]
    finally:
        conn.close()