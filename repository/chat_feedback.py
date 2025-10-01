from utils import logger
from utils.database import get_session
from utils.time import now_kst, to_kst_string
from sqlalchemy import select, desc, func
from sqlalchemy.exc import IntegrityError
from storage.db_models import ChatFeedback, WorkspaceChat, SystemPromptTemplate
from errors import DatabaseError

logger = logger(__name__)


def get_chat_by_id(chat_id: int, user_id: int) -> dict | None:
    """채팅 메시지 조회"""
    with get_session() as session:
        stmt = select(
            WorkspaceChat.id,
            WorkspaceChat.category,
            WorkspaceChat.prompt,
            WorkspaceChat.response,
            WorkspaceChat.workspace_id,
            WorkspaceChat.thread_id,
        ).where(
            WorkspaceChat.id == chat_id,
            WorkspaceChat.user_id == user_id,
        )
        row = session.execute(stmt).first()
        if not row:
            return None
        
        m = row._mapping
        return {
            "id": m["id"],
            "category": m["category"],
            "prompt": m["prompt"],
            "response": m["response"],
            "workspace_id": m["workspace_id"],
            "thread_id": m["thread_id"],
        }


def save_feedback_metadata(
    category: str,
    subcategory: str | None,
    filename: str,
    file_path: str,
    prompt_id: int | None,
) -> int:
    """피드백 메타데이터를 DB에 저장"""
    with get_session() as session:
        try:
            obj = ChatFeedback(
                category=category,
                subcategory=subcategory,
                filename=filename,
                file_path=file_path,
                prompt_id=prompt_id,
                created_at=now_kst(),
                updated_at=now_kst(),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            logger.info(f"save_feedback_metadata success: id={obj.id}, file={filename}")
            return obj.id
        except IntegrityError as exc:
            logger.error(f"save_feedback_metadata failed: {exc}")
            raise DatabaseError(str(exc))


def update_feedback_metadata(feedback_id: int) -> None:
    """피드백 메타데이터 업데이트 (updated_at 갱신)"""
    with get_session() as session:
        stmt = select(ChatFeedback).where(ChatFeedback.id == feedback_id)
        feedback = session.execute(stmt).scalar_one_or_none()
        if feedback:
            feedback.updated_at = now_kst()
            session.commit()


def get_feedback_by_file_info(
    category: str,
    subcategory: str | None,
    prompt_id: int | None
) -> dict | None:
    """동일한 파일 정보로 기존 피드백 레코드 조회"""
    with get_session() as session:
        stmt = select(ChatFeedback).where(
            ChatFeedback.category == category,
            ChatFeedback.prompt_id == prompt_id,
        )
        if subcategory:
            stmt = stmt.where(ChatFeedback.subcategory == subcategory)
        else:
            stmt = stmt.where(ChatFeedback.subcategory.is_(None))
        
        row = session.execute(stmt).first()
        if not row:
            return None
        
        feedback = row[0]
        return {
            "id": feedback.id,
            "category": feedback.category,
            "subcategory": feedback.subcategory,
            "filename": feedback.filename,
            "file_path": feedback.file_path,
            "prompt_id": feedback.prompt_id,
            "created_at": to_kst_string(feedback.created_at),
            "updated_at": to_kst_string(feedback.updated_at),
        }


def list_all_feedbacks(
    category: str | None = None,
    prompt_id: int | None = None
) -> list[dict]:
    """피드백 파일 목록 조회"""
    with get_session() as session:
        stmt = select(ChatFeedback).order_by(desc(ChatFeedback.updated_at))
        
        if category:
            stmt = stmt.where(ChatFeedback.category == category)
        if prompt_id:
            stmt = stmt.where(ChatFeedback.prompt_id == prompt_id)
        
        rows = session.execute(stmt).scalars().all()
        
        result = []
        for feedback in rows:
            result.append({
                "id": feedback.id,
                "category": feedback.category,
                "subcategory": feedback.subcategory,
                "filename": feedback.filename,
                "file_path": feedback.file_path,
                "prompt_id": feedback.prompt_id,
                "prompt_name": feedback.prompt_template.name if feedback.prompt_template else None,
                "created_at": to_kst_string(feedback.created_at),
                "updated_at": to_kst_string(feedback.updated_at),
            })
        
        return result