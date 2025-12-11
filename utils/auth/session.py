"""세션 인증 미들웨어 """

from fastapi import Cookie, HTTPException
from repository.session import get_session_from_db
from errors import SessionNotFoundError
from utils import logger

logger = logger(__name__)

def get_user_id_from_cookie(coreiq_session: str = Cookie(None)) -> int:
    """쿠키에서 사용자 ID 추출"""
    if not coreiq_session:
        raise SessionNotFoundError("로그인이 필요합니다")
    session_data = get_session_from_db(coreiq_session)
    if not session_data:
        raise SessionNotFoundError("유효하지 않은 세션입니다")
    logger.info(f"get_user_id_from_cookie: {session_data['user_id']}")
    return session_data['user_id']

def get_user_info_from_cookie(coreiq_session: str = Cookie(None)) -> dict:
    """쿠키에서 사용자 전체 정보 추출"""
    if not coreiq_session:
        raise SessionNotFoundError("로그인이 필요합니다")
    session_data = get_session_from_db(coreiq_session)
    if not session_data:
        raise SessionNotFoundError("유효하지 않은 세션입니다")
    return session_data

