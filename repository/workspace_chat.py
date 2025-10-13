from utils import logger
from utils.database import get_session
from utils.time import now_kst, to_kst_string
from datetime import datetime
from sqlalchemy import select, desc
from sqlalchemy.exc import IntegrityError
from storage.db_models import WorkspaceChat
from errors import DatabaseError

logger = logger(__name__)


def get_chat_history_by_workspace_id(
    user_id: int, workspace_id: int, limit: int | None = None
) -> list[dict]:
    with get_session() as session:
        stmt = (
            select(
                WorkspaceChat.id,
                WorkspaceChat.category,
                WorkspaceChat.thread_id,
                WorkspaceChat.prompt,
                WorkspaceChat.response,
                WorkspaceChat.created_at,
            )
            .where(
                WorkspaceChat.user_id == user_id,
                WorkspaceChat.workspace_id == workspace_id,
            )
            .order_by(desc(WorkspaceChat.created_at))
        )
        if limit and limit > 0:
            stmt = stmt.limit(int(limit))
        rows = session.execute(stmt).all()
        if not rows:
            return []
        items: list[dict] = []
        for row in rows:
            m = row._mapping
            created = m["created_at"]
            items.append(
                {
                    "id": m["id"],
                    "category": m["category"],
                    "thread_id": m["thread_id"],
                    "prompt": m["prompt"],
                    "response": m["response"],
                    "created_at": to_kst_string(created),
                }
            )
        return items


def get_chat_history_by_thread_id(
    user_id: int, thread_id: int, limit: int
) -> list[dict]:
    with get_session() as session:
        stmt = (
            select(
                WorkspaceChat.id,
                WorkspaceChat.prompt,
                WorkspaceChat.response,
                WorkspaceChat.created_at,
            )
            .where(
                WorkspaceChat.user_id == user_id,
                WorkspaceChat.thread_id == thread_id,
            )
            .order_by(desc(WorkspaceChat.created_at))
            .limit(int(limit))
        )
        rows = session.execute(stmt).all()
        if not rows:
            return []
        items: list[dict] = []
        for row in rows:
            m = row._mapping
            created = m["created_at"]
            items.append(
                {
                    "id": m["id"],
                    "prompt": m["prompt"],
                    "response": m["response"],
                    "created_at": to_kst_string(created),
                }
            )
        return items


def insert_chat_history(
    user_id: int,
    category: str,
    workspace_id: int,
    prompt: str,
    response: str,
    thread_id: int | None = None,
):
    with get_session() as session:
        try:
            obj = WorkspaceChat(
                user_id=user_id,
                category=category,
                workspace_id=workspace_id,
                prompt=prompt,
                response=response,
                thread_id=thread_id,
                created_at=now_kst(),
                updated_at=now_kst(),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            return logger.info(f"insert_chat_history success: id={obj.id}")
        except IntegrityError as exc:
            logger.error(f"insert_chat_history failed: {exc}")
            raise DatabaseError(str(exc))
