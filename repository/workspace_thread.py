from utils import logger
from utils.database import get_session
from utils.time import now_kst, to_kst_string
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from storage.db_models import WorkspaceThread
from errors import DatabaseError
from repository.workspace import get_workspace_id_by_name

logger = logger(__name__)

###


def create_default_thread(
    user_id: int, name: str, thread_slug: str, workspace_id: int = None
) -> int:
    """워크스페이스의 기본 스레드를 생성합니다."""
    with get_session() as session:
        try:
            if not workspace_id:
                workspace_id = get_workspace_id_by_name(user_id, name)
            if not workspace_id:
                raise DatabaseError("workspace not found for thread creation")

            obj = WorkspaceThread(
                user_id=user_id,
                name=name,
                slug=thread_slug,
                workspace_id=workspace_id,
                created_at=now_kst(),
                updated_at=now_kst(),
            )
            session.add(obj)
            session.commit()
            session.refresh(obj)
            thread_id = int(obj.id)
            logger.info(
                f"Default thread created: id={thread_id}, workspace_id={workspace_id}"
            )
            return thread_id
        except IntegrityError as exc:
            logger.error(f"thread creation failed: {exc}")
            session.rollback()
            raise DatabaseError(f"thread creation failed: {exc}") from exc


def get_thread_by_id(thread_id: int) -> dict | None:
    with get_session() as session:
        stmt = select(
            WorkspaceThread.id,
            WorkspaceThread.name,
            WorkspaceThread.slug,
            WorkspaceThread.user_id,
            WorkspaceThread.workspace_id,
        ).where(WorkspaceThread.id == thread_id)
        row = session.execute(stmt).first()
        if row:
            logger.debug(f"Thread fetched: id={thread_id}")
            return dict(row._mapping)
        else:
            logger.warning(f"Thread not found: id={thread_id}")
            return None


def get_threads_by_workspace_id(workspace_id: int) -> list[dict]:
    with get_session() as session:
        stmt = select(
            WorkspaceThread.id,
            WorkspaceThread.name,
            WorkspaceThread.slug,
            WorkspaceThread.user_id,
            WorkspaceThread.workspace_id,
            WorkspaceThread.created_at,
            WorkspaceThread.updated_at,
        ).where(WorkspaceThread.workspace_id == workspace_id)
        rows = session.execute(stmt).all()
        if rows:
            logger.debug(f"Threads fetched: workspace_id={workspace_id}")
        else:
            logger.warning(f"No threads found for workspace_id={workspace_id}")
        items: list[dict] = []
        for row in rows:
            m = dict(row._mapping)
            items.append(m)
        return items


def get_thread_id_by_slug_for_user(user_id: int, thread_slug: str) -> int | None:
    with get_session() as session:
        stmt = (
            select(WorkspaceThread.id)
            .where(
                WorkspaceThread.user_id == user_id,
                WorkspaceThread.slug == thread_slug,
            )
            .limit(1)
        )
        return session.execute(stmt).scalar()


def get_thread_by_slug_for_user(user_id: int, thread_slug: str) -> dict | None:
    with get_session() as session:
        stmt = (
            select(
                WorkspaceThread.id,
                WorkspaceThread.name,
                WorkspaceThread.slug,
                WorkspaceThread.user_id,
                WorkspaceThread.workspace_id,
            )
            .where(
                WorkspaceThread.user_id == user_id,
                WorkspaceThread.slug == thread_slug,
            )
            .limit(1)
        )
        row = session.execute(stmt).first()
        return dict(row._mapping) if row else None


def update_thread_name_by_slug_for_user(
    user_id: int, thread_slug: str, new_name: str
) -> dict | None:
    with get_session() as session:
        try:
            now_kst_string = now_kst()
            stmt = (
                update(WorkspaceThread)
                .where(
                    WorkspaceThread.user_id == user_id,
                    WorkspaceThread.slug == thread_slug,
                )
                .values(name=new_name, updated_at=now_kst_string)
            )
            result = session.execute(stmt)
            session.commit()
            if result.rowcount == 0:
                return None
            # 업데이트된 행 조회
            stmt_sel = (
                select(
                    WorkspaceThread.id,
                    WorkspaceThread.name,
                    WorkspaceThread.slug,
                    WorkspaceThread.user_id,
                    WorkspaceThread.workspace_id,
                )
                .where(
                    WorkspaceThread.user_id == user_id,
                    WorkspaceThread.slug == thread_slug,
                )
                .limit(1)
            )
            row = session.execute(stmt_sel).first()
            return dict(row._mapping) if row else None
        except IntegrityError as exc:
            logger.error(f"thread update failed: {exc}")
            session.rollback()
            raise DatabaseError(f"thread update failed: {exc}") from exc
