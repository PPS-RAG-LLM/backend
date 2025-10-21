import json
from errors.exceptions import NotFoundError
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
                WorkspaceChat.model,
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
                    "model": m["model"],
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
                WorkspaceChat.model,
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
                    "model": m["model"],
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
    model: str | None = None,
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
                model=model,
                created_at=now_kst(),
                updated_at=now_kst(),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            logger.info(f"insert_chat_history success: id={obj.id}")
            return obj.id # id 반환
        except IntegrityError as exc:
            logger.error(f"insert_chat_history failed: {exc}")
            raise DatabaseError(str(exc))

def update_chat_metrics(chat_id: int, user_id: int, reasoning_duration: float) -> None:
    """채팅 메시지의 reasoning_duration 업데이트"""
    with get_session() as session:
        try:
            chat = session.query(WorkspaceChat).filter(
                WorkspaceChat.id == chat_id,
                WorkspaceChat.user_id == user_id
            ).first()
            
            if not chat:
                raise NotFoundError(f"Chat not found: chat_id={chat_id}")
            
            # response JSON 파싱
            response_data = json.loads(chat.response)
            
            # metrics에 reasoning_duration 업데이트
            if "metrics" in response_data:
                response_data["metrics"]["reasoning_duration"] = round(reasoning_duration, 3)
            else:
                response_data["metrics"] = {"reasoning_duration": round(reasoning_duration, 3)}
            
            # 업데이트
            chat.response = json.dumps(response_data, ensure_ascii=False)
            chat.updated_at = now_kst()
            session.commit()
            
            logger.info(f"update_chat_metrics success: chat_id={chat_id}, reasoning_duration={reasoning_duration}")
        except Exception as exc:
            session.rollback()
            logger.error(f"update_chat_metrics failed: {exc}")
            raise DatabaseError(str(exc))