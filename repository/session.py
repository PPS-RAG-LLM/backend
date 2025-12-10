from utils import logger
from utils.database import get_session
from typing import Optional
from storage.db_models import UserSession, User
from sqlalchemy import select, delete
from utils.time import now_kst, to_kst_string
from datetime import timedelta, datetime

logger = logger(__name__)


def create_new_session(user_id: int) -> str:
    """새 세션 생성 및 저장"""
    import secrets

    session_id = secrets.token_urlsafe(32)
    save_session_to_db(session_id, user_id)
    return session_id


# 메모리 세션 대신 DB 세션 사용
def save_session_to_db(session_id: str, user_id: int):
    """세션을 DB에 저장"""
    with get_session() as session:
        time_kst = now_kst()
        expires_at_kst = time_kst + timedelta(hours=8)  # 8시간 유효
        obj = UserSession(
            session_id=session_id,
            user_id=int(user_id),
            created_at=time_kst,
            expires_at=expires_at_kst,
        )
        # 같은 키가 있으면 교체 동작을 원한다면 merge 사용 가능
        session.merge(obj)
        session.commit()


def _parse_db_datetime(value) -> Optional[datetime]:
    """DB 값을 datetime으로 변환 (문자열인 경우 파싱)"""
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None
    return None


def get_session_from_db(session_id: str) -> Optional[dict]:
    """세션 조회 (users 테이블과 JOIN해서 사용자 정보도 함께)"""
    with get_session() as session:
        stmt = (
            select(
                User.id,
                User.role,
                User.username,
                User.name,
                User.department,
                User.position,
                User.security_level,
                UserSession.expires_at,
            )
            .join(User, User.id == UserSession.user_id)
            .where(UserSession.session_id == session_id)
            .limit(1)
        )
        row = session.execute(stmt).first()
        if not row:
            return None

        user_id, role, username, name, department, position, security_level, expires_at = row

        # 만료 체크
        expires_dt = _parse_db_datetime(expires_at)
        
        now_dt = now_kst()
        if not expires_dt or expires_dt <= now_dt:
            # 만료된 세션 삭제
            session.execute(
                delete(UserSession).where(UserSession.session_id == session_id)
            )
            session.commit()
            logger.info(f"delete expired session: {session_id}")
            return None

        return {
            "user_id": user_id,
            "role": role,
            "username": username,
            "name": name,
            "department": department,
            "position": position,
            "security_level": security_level,
            "expires_at": to_kst_string(expires_dt),  # KST 문자열로 반환
        }


def list_all_sessions_from_db() -> list[dict]:
    """DB에 저장된 모든 세션 정보를 조회합니다."""
    with get_session() as session:
        stmt = select(
            UserSession.session_id,
            User.id,
            User.username,
            User.name,
            User.department,
            User.position,
            User.security_level,
            UserSession.created_at,
            UserSession.expires_at,
        ).join(User, User.id == UserSession.user_id)
        rows = session.execute(stmt).all()
        items = []
        for row in rows:
            (
                session_id,
                user_id,
                username,
                name,
                department,
                position,
                security_level,
                created_at,
                expires_at,
            ) = row
            created_str = (
                to_kst_string(created_at)
                if isinstance(created_at, datetime)
                else str(created_at)
            )
            
            exp_dt = _parse_db_datetime(expires_at)
            exp_str = to_kst_string(exp_dt) if exp_dt else str(expires_at)

            items.append(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "username": username,
                    "name": name,
                    "department": department,
                    "position": position,
                    "security_level": security_level,
                    "created_at": created_str,
                    "expires_at": exp_str,
                }
            )
        return items


def delete_session_from_db(session_id: str) -> bool:
    """DB에서 세션 삭제"""
    with get_session() as session:
        result = session.execute(
            delete(UserSession).where(UserSession.session_id == session_id)
        )
        session.commit()
        # 삭제 확인
        return result.rowcount > 0


def delete_sessions_by_user_id(user_id: int):
    """사용자 ID에 해당하는 모든 세션 삭제"""
    with get_session() as session:
        session.execute(delete(UserSession).where(UserSession.user_id == int(user_id)))
        session.commit()
        logger.info(f"delete sessions by user_id: {user_id}")


def cleanup_expired_sessions() -> int:
    """만료된 모든 세션 일괄 정리 (레거시 문자열/신규 datetime 모두 처리)"""
    with get_session() as session:
        now_dt = now_kst()
        # 한 번에 지우기 어려운 레거시 문자열을 고려, 일괄 조회 후 개별 삭제
        stmt = select(UserSession.session_id, UserSession.expires_at)
        rows = session.execute(stmt).all()
        expired_session_ids = []
        for session_id, expires_at in rows:
            exp_dt = _parse_db_datetime(expires_at)
            # 파싱 실패 시 안전하게 삭제하지 않음 (또는 정책 결정)
            if exp_dt and exp_dt <= now_dt:
                expired_session_ids.append(session_id)
        if expired_session_ids:
            session.execute(
                delete(UserSession).where(
                    UserSession.session_id.in_(expired_session_ids)
                )
            )
            session.commit()
            logger.info(
                f"clean up expired sessions: {len(expired_session_ids)} sessions"
            )
        return len(expired_session_ids)
