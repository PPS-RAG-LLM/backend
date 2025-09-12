from typing import Optional, Dict, Any
import sqlite3
from utils import logger, get_db, now_kst_string
from errors import DatabaseError

logger = logger(__name__)

### 
def get_default_llm_model(category: str) -> Optional[Dict[str, str]]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT provider, name
            FROM llm_models
            WHERE category=? AND is_default=1 AND is_active=1
            ORDER BY id DESC LIMIT 1
            """,
            (category,),
        )
        row = cur.fetchone()
        if not row:
            logger.warning(f"No default llm model for category={category}")
            return None
        model = {"provider": row["provider"], "chat_model": row["name"]}
        logger.debug(f"Default LLM model selected: {model}")
        return model
    finally:
        conn.close()

def insert_workspace(
    *,
    name: str,
    slug: str,
    category: str,
    temperature: Optional[float],
    chat_history: int,
    system_prompt: Optional[str],
    similarity_threshold: Optional[float],
    provider: str,
    chat_model: str,
    top_n: int,
    chat_mode: str,
    query_refusal_response: Optional[str],
) -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO workspaces (
                name, slug, category, temperature, chat_history, system_prompt,
                similarity_threshold, provider, chat_model, top_n, chat_mode, query_refusal_response,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name, slug, category, temperature, chat_history, system_prompt,
                similarity_threshold, provider, chat_model, top_n, chat_mode, query_refusal_response,
                now_kst_string(), now_kst_string()
            ),
        )
        conn.commit()
        workspace_id = cur.lastrowid
        logger.info(f"Workspace inserted: id={workspace_id}, slug={slug}, category={category}")
        return workspace_id
    except sqlite3.IntegrityError as exc:
        logger.error(f"Workspace insert failed: {exc}")
        raise DatabaseError(f"workspace insert failed: {exc}") from exc
    finally:
        conn.close()

def get_workspace_id_by_name(user_id: int, name: str) -> int:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id FROM workspaces WHERE name=?
            """,
            (name,),
        )
        row = cur.fetchone()
        if row:
            logger.debug(f"Workspace fetched: id={row['id']}")
        else:
            logger.warning(f"Workspace not found: name={name}")
        return row["id"] if row else None
    finally:
        conn.close()

### 
def get_workspace_by_id(workspace_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              id,
              name,
              slug,
              category,
              created_at,
              updated_at,
              temperature,
              chat_history,
              system_prompt
            FROM workspaces
            WHERE id=?
            """,
            (workspace_id,),
        )
        row = cur.fetchone()
        if row:
            logger.debug(f"Workspace fetched: id={workspace_id}")
        else:
            logger.warning(f"Workspace not found: id={workspace_id}")
        return dict(row) if row else None
    finally:
        conn.close()


def link_workspace_to_user(user_id: int, workspace_id: int) -> None:
    """유저가 워크스페이스 생성 시 : workspace_users 테이블에 유저와 워크스페이스 연결"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO workspace_users (user_id, workspace_id, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, workspace_id, now_kst_string(), now_kst_string()),
        )
        conn.commit()
        logger.debug(f"Workspace linked to user: user_id={user_id}, workspace_id={workspace_id}")
    finally:
        conn.close()


def get_default_system_prompt_content(category: str) -> Optional[str]:
    """기본값 프롬프트 조회"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT content
            FROM system_prompt_template
            WHERE category = ? AND is_default = 1 AND is_active = 1
            ORDER BY id DESC
            LIMIT 1
            """,
            (category,),
        )
        row = cur.fetchone()
        if not row:
            logger.warning(f"No default system prompt for category={category}")
            return None
        return row["content"]
    finally:
        conn.close()

###############################



def get_workspaces_by_user(user_id: int) -> list[Dict[str, Any]]:
    con = get_db()
    try:
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT
              w.id,
              w.name,
              w.slug,
              w.category,
              w.created_at,
              w.updated_at,
              w.temperature,
              w.chat_history,
              w.system_prompt
            FROM workspaces AS w
            INNER JOIN workspace_users AS wu ON wu.workspace_id = w.id
            WHERE wu.user_id = ?
            ORDER BY w.id DESC
            """,
            (user_id,)
        )
        rows = rows.fetchall()
        if not rows:
            return []
        return rows
    finally:
        con.close()


def get_workspace_by_workspace_id(user_id: int, workspace_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              w.id,
              w.name,
              w.slug,
              w.category,
              w.created_at,
              w.updated_at,
              w.temperature,
              w.chat_history,
              w.system_prompt,
              w.similarity_threshold,
              w.provider,
              w.chat_model,
              w.top_n,
              w.chat_mode,
              w.query_refusal_response
            FROM workspaces AS w
            INNER JOIN workspace_users AS wu ON wu.workspace_id = w.id
            WHERE w.id = ? AND wu.user_id = ?
            LIMIT 1
            """,
            (workspace_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_workspace_by_slug_for_user(user_id: int, slug: str) -> bool:
    con = get_db()
    try:
        cur = con.cursor()
        cur.execute(
            """
            DELETE FROM workspaces
            WHERE id IN (
                SELECT w.id
                FROM workspaces AS w
                INNER JOIN workspace_users AS wu ON wu.workspace_id = w.id
                WHERE w.slug = ? AND wu.user_id = ?
            )
            """,
            (slug, user_id),
        )
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()

def get_workspace_id_by_slug_for_user(user_id: int, slug: str) -> Optional[int]:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT w.id
            FROM workspaces AS w
            INNER JOIN workspace_users AS wu ON wu.workspace_id = w.id
            WHERE w.slug = ? AND wu.user_id = ?
            LIMIT 1
            """,
            (slug, user_id),
        )
        row = cur.fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def update_workspace_by_slug_for_user(user_id: int, slug: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """선택적 필드만 업데이트하고, 갱신된 행을 반환한다."""
    # 매핑: API 키 -> DB 컬럼
    key_to_col = {
        "name": "name",
        "temperature": "temperature",
        "chatHistory": "chat_history",
        "systemPrompt": "system_prompt",
        "slug": "slug",
    }
    set_parts = []
    params = []
    for key, col in key_to_col.items():
        if key in updates:
            set_parts.append(f"{col} = ?")
            params.append(updates[key])

    # updated_at은 항상 갱신
    set_clause = ", ".join(set_parts + ["updated_at = ?"])
    updated_at_value = now_kst_string()
    logger.info(f"updated_at_value: {updated_at_value}")

    con = get_db()
    try:
        cur = con.cursor()
        # 권한 검증과 함께 업데이트 수행
        sql = f"""
            UPDATE workspaces
            SET {set_clause}
            WHERE id IN (
                SELECT w.id
                FROM workspaces AS w
                INNER JOIN workspace_users AS wu ON wu.workspace_id = w.id
                WHERE w.slug = ? AND wu.user_id = ?
            )
        """
        cur.execute(sql, (*params, updated_at_value, slug, user_id))
        con.commit()
        logger.info(f"updated workspace.")
        if cur.rowcount == 0:
            logger.error(f"workspace update failed: {cur.rowcount}")
            raise DatabaseError(f"workspace update failed: {cur.rowcount}")
    except sqlite3.IntegrityError as exc:
        logger.error(f"workspace update failed: {exc}")
        raise DatabaseError(f"workspace update failed: {exc}") from exc
    finally:
        con.close()


