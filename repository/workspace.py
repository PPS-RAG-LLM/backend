from typing import Optional, Dict, Any
from datetime import datetime
from utils import logger
from utils.database import get_session
from utils.time import to_kst_string, now_kst_string, now_kst
from storage.db_models import (
    Document,
    DocumentMetadata,
    DocumentType,
    Workspace,
    WorkspaceUser,
    LlmModel,
    SystemPromptTemplate,
)
from sqlalchemy import select, update, delete, func
from errors import DatabaseError

logger = logger(__name__)


###
def get_default_llm_model(category: str) -> Optional[Dict[str, str]]:
    """카테고리별 기본 LLM 모델 조회"""
    with get_session() as session:

        def _pick(cat: str):
            stmt = (
                select(LlmModel.provider, LlmModel.name)
                .where(
                    LlmModel.category == cat,
                    LlmModel.is_default == True,
                    LlmModel.is_active == True,
                )
                .order_by(LlmModel.id.desc())
                .limit(1)
            )
            return session.execute(stmt).first()

        result = _pick(category) or _pick("all")
        if not result:
            logger.warning(f"No default llm model for category={category}")
            return None

        model = {"provider": result.provider, "chat_model": result.name}
        logger.debug(f"Default LLM model selected: {model}")
        return model


def insert_workspace(
    *,
    name                : str,
    slug                : str,
    category            : str,
    temperature         : Optional[float],
    chat_history        : int,
    system_prompt       : Optional[str],
    similarity_threshold: Optional[float],
    provider            : Optional[str],
    top_n               : int,
    chat_mode           : Optional[str],
    query_refusal_response: Optional[str],
    vector_search_mode  : Optional[str],
) -> int:
    """워크스페이스 생성"""
    with get_session() as session:
        try:
            now_dt = now_kst()
            workspace = Workspace(
                name                    = name,
                slug                    = slug,
                category                = category,
                temperature             = temperature,
                chat_history            = chat_history,
                system_prompt           = system_prompt,
                provider                = provider,
                similarity_threshold    = similarity_threshold,
                top_n                   = top_n,
                chat_mode               = chat_mode,
                query_refusal_response  = query_refusal_response,
                vector_search_mode      = vector_search_mode,
                created_at              = now_dt,
                updated_at              = now_dt,
            )
            session.add(workspace)
            session.commit()
            session.refresh(workspace)

            logger.info(
                f"Workspace inserted: id={workspace.id}, slug={slug}, category={category}"
            )
            return workspace.id

        except Exception as exc:
            session.rollback()
            logger.error(f"Workspace insert failed: {exc}")
            raise DatabaseError(f"workspace insert failed: {exc}") from exc


def get_workspace_id_by_name(user_id: int, name: str) -> Optional[int]:
    """워크스페이스 이름으로 ID 조회"""
    with get_session() as session:
        stmt = select(Workspace.id).where(Workspace.name == name)
        result = session.execute(stmt).scalar()

        if result:
            logger.debug(f"Workspace fetched: id={result}")
        else:
            logger.warning(f"Workspace not found: name={name}")

        return result


###
def get_workspace_by_id(workspace_id: int) -> Optional[Dict[str, Any]]:
    """워크스페이스 ID로 기본 정보 조회"""
    with get_session() as session:
        stmt = select(
            Workspace.id,
            Workspace.name,
            Workspace.slug,
            Workspace.category,
            Workspace.created_at,
            Workspace.updated_at,
            Workspace.temperature,
            Workspace.chat_history,
            Workspace.system_prompt,
            Workspace.provider,
            Workspace.vector_search_mode,
        ).where(Workspace.id == workspace_id)

        result = session.execute(stmt).first()
        if result:
            logger.debug(f"Workspace fetched: id={workspace_id}")
            m = dict(result._mapping)
            # datetime -> KST 문자열 변환
            if isinstance(m.get("created_at"), datetime):
                m["created_at"] = to_kst_string(m["created_at"])
            if isinstance(m.get("updated_at"), datetime):
                m["updated_at"] = to_kst_string(m["updated_at"])
            return m
        else:
            logger.warning(f"Workspace not found: id={workspace_id}")
            return None


def link_workspace_to_user(user_id: int, workspace_id: int) -> None:
    """유저가 워크스페이스 생성 시 : workspace_users 테이블에 유저와 워크스페이스 연결"""
    with get_session() as session:
        # 이미 연결된 경우 무시 (INSERT OR IGNORE 효과)
        existing = (
            session.query(WorkspaceUser)
            .filter(
                WorkspaceUser.user_id == user_id,
                WorkspaceUser.workspace_id == workspace_id,
            )
            .first()
        )

        if not existing:
            now_dt = now_kst()
            workspace_user  = WorkspaceUser(
                user_id     = user_id,
                workspace_id= workspace_id,
                created_at  = now_dt,
                updated_at  = now_dt,
            )
            session.add(workspace_user)
            session.commit()

        logger.debug(
            f"Workspace linked to user: user_id={user_id}, workspace_id={workspace_id}"
        )


def get_default_system_prompt_content(category: str) -> Optional[str]:
    """기본값 프롬프트 조회"""
    with get_session() as session:
        stmt = (
            select(SystemPromptTemplate.system_prompt)
            .where(
                SystemPromptTemplate.category   == category,
                SystemPromptTemplate.is_default == True,
                SystemPromptTemplate.is_active  == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
            .limit(1)
        )

        result = session.execute(stmt).scalar()
        if not result:
            logger.warning(f"No default system prompt for category={category}")
            return None

        return result


###############################


def get_workspaces_by_user(user_id: int) -> list[Dict[str, Any]]:
    """유저별 워크스페이스 목록 조회"""
    with get_session() as session:
        stmt = (
            select(
                Workspace.id,
                Workspace.name,
                Workspace.slug,
                Workspace.provider,
                Workspace.category,
                Workspace.created_at,
                Workspace.updated_at,
                Workspace.temperature,
                Workspace.chat_history,
                Workspace.system_prompt,
            )
            .join(WorkspaceUser)
            .where(WorkspaceUser.user_id == user_id)
            .order_by(Workspace.id.desc())
        )

        results = session.execute(stmt).all()
        if not results:
            return []

        items = [dict(row._mapping) for row in results]
        # datetime -> KST 문자열 변환
        for m in items:
            if isinstance(m.get("created_at"), datetime):
                m["created_at"] = to_kst_string(m["created_at"])
            if isinstance(m.get("updated_at"), datetime):
                m["updated_at"] = to_kst_string(m["updated_at"])
        return items

def update_workspace_vector_count(workspace_id: int) -> int:
    """
    워크스페이스의 총 벡터 수를 계산하여 workspace.vector_count에 업데이트.
    Returns: 업데이트된 벡터 수
    """
    with get_session() as session:
        stmt = (
            select(Document.doc_id, Document.payload)
            .where(
                Document.workspace_id == workspace_id,
                Document.doc_type == DocumentType.WORKSPACE.value,
            )
        )
        rows = session.execute(stmt).all()

        total_chunks = 0
        missing: list[str] = []
        for doc_id, payload in rows:
            meta = (payload or {}).get("workspace_metadata") or {}
            chunks = meta.get("chunks")
            try:
                value = int(chunks)
            except (TypeError, ValueError):
                missing.append(doc_id)
            else:
                total_chunks += value

        if missing:
            count_stmt = (
                select(DocumentMetadata.doc_id, func.count())
                .where(DocumentMetadata.doc_id.in_(missing))
                .group_by(DocumentMetadata.doc_id)
            )
            for _, count in session.execute(count_stmt).all():
                total_chunks += int(count or 0)

        stmt_update = (
            update(Workspace)
            .where(Workspace.id == workspace_id)
            .values(vector_count=total_chunks)
        )
        session.execute(stmt_update)
        session.commit()

        return total_chunks

def get_workspace_by_workspace_id(
    user_id: int, workspace_id: int
) -> Optional[Dict[str, Any]]:
    """유저 권한 확인 후 워크스페이스 상세 정보 조회"""
    with get_session() as session:
        stmt = (
            select(
                Workspace.id,
                Workspace.name,
                Workspace.slug,
                Workspace.category,
                Workspace.created_at,
                Workspace.updated_at,
                Workspace.temperature,
                Workspace.chat_history,
                Workspace.system_prompt,
                Workspace.similarity_threshold,
                Workspace.provider,
                Workspace.chat_model,
                Workspace.top_n,
                Workspace.chat_mode,
                Workspace.query_refusal_response,
                Workspace.similarity_threshold,
                Workspace.vector_search_mode,
                Workspace.vector_count,
            )
            .join(WorkspaceUser)
            .where(Workspace.id == workspace_id, WorkspaceUser.user_id == user_id)
            .limit(1)
        )

        result = session.execute(stmt).first()
        if not result:
            return None
        m = dict(result._mapping)
        if isinstance(m.get("created_at"), datetime):
            m["created_at"] = to_kst_string(m["created_at"])
        if isinstance(m.get("updated_at"), datetime):
            m["updated_at"] = to_kst_string(m["updated_at"])
        return m


def delete_workspace_by_workspace_id(workspace_id: int) -> bool:
    """유저 권한 확인 후 워크스페이스 삭제"""
    with get_session() as session:
        # 유저에게 권한이 있는 워크스페이스 찾기
        workspace_id_subquery = (
            select(WorkspaceUser.workspace_id)
            .join(Workspace)
            .where(Workspace.id == workspace_id)
        )
        # 워크스페이스 삭제
        stmt = delete(Workspace).where(Workspace.id.in_(workspace_id_subquery))
        result = session.execute(stmt)
        session.commit()
        return result.rowcount > 0


def get_workspace_id_by_slug_for_user(user_id: int, slug: str) -> Optional[int]:
    """유저 권한 확인 후 워크스페이스 ID 조회"""
    with get_session() as session:
        stmt = (
            select(Workspace.id)
            .join(WorkspaceUser)
            .where(Workspace.slug == slug, WorkspaceUser.user_id == user_id)
            .limit(1)
        )

        result = session.execute(stmt).scalar()
        return result

def update_workspace_name_by_slug_for_user(user_id:int, slug: str, name: str) -> Optional[Dict[str, Any]]:
    """워크스페이스 이름 업데이트"""
    with get_session() as session:
        workspace_id_subquery = (
            select(WorkspaceUser.workspace_id)
            .where(WorkspaceUser.user_id == user_id)
            .join(Workspace)
            .where(Workspace.slug == slug)
        )
        stmt = update(Workspace).where(Workspace.id.in_(workspace_id_subquery)).values(name=name)
        session.execute(stmt)
        session.commit()
        return None

def update_workspace_by_slug_for_user(
    user_id: int, slug: str, updates: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """선택적 필드만 업데이트하고, 갱신된 행을 반환한다."""
    # 매핑: API 키 -> DB 컬럼
    key_to_col = {
        "temperature": "temperature",
        "chatHistory": "chat_history",
        "systemPrompt": "system_prompt",
        "provider":"provider",
        "vectorSearchMode":"vector_search_mode",
        "similarityThreshold":"similarity_threshold",
        "topN":"top_n",
        "queryRefusalResponse":"query_refusal_response",
    }

    # 업데이트할 필드만 추출
    update_data = {}
    for key, col in key_to_col.items():
        if key in updates:
            update_data[col] = updates[key]

    if not update_data:
        return None

    # updated_at은 항상 갱신 (DB에는 UTC 저장, 응답 시 KST 문자열 변환)
    update_data["updated_at"] = now_kst()
    updated_at_value = now_kst_string()
    logger.info(f"updated_at_value: {updated_at_value}")

    with get_session() as session:
        try:
            # 유저에게 권한이 있는 워크스페이스만 업데이트
            workspace_id_subquery = (
                select(WorkspaceUser.workspace_id)
                .where(WorkspaceUser.user_id == user_id)
                .join(Workspace)
                .where(Workspace.slug == slug)
            )

            stmt = (
                update(Workspace)
                .where(Workspace.id.in_(workspace_id_subquery))
                .values(**update_data)
            )

            result = session.execute(stmt)
            session.commit()

            logger.info(f"updated workspace.")
            if result.rowcount == 0:
                logger.error(f"workspace update failed: {result.rowcount}")
                raise DatabaseError(f"workspace update failed: {result.rowcount}")

        except Exception as exc:
            session.rollback()
            logger.error(f"workspace update failed: {exc}")
            raise DatabaseError(f"workspace update failed: {exc}") from exc
