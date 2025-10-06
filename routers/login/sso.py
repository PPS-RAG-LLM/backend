# routers/sso.py (완전 버전)
from fastapi import APIRouter, Response, Cookie
from pydantic import BaseModel
from utils import get_db, now_kst_string
from typing import Optional, Dict, Any
from errors import SessionNotFound, NotFoundError, UnauthorizedError
from service.users.session import create_session
from utils import logger
from config import config
from utils.time import now_kst

logger = logger(__name__)

sso_router = APIRouter(prefix="/v1/sso", tags=["SSO"])

class CompanyUserInfo(BaseModel):
    username: str      # 회사의 employee_id
    password: str

@sso_router.post("/login")
def sso_login(company_user: CompanyUserInfo, response: Response):
    """회사에서 넘어온 ID/PW로 로그인 및 세션 생성"""
    try:
        server_conf = config.get("server")
        # 1. 로그인 처리 및 세션 생성
        session_id, user_info = create_session(company_user.username, company_user.password)
        # 2. 브라우저에 쿠키 설정
        response.set_cookie(
            key     =server_conf.get("cookie_name").lower(),
            value   =session_id,
            max_age =server_conf.get("cookie_session_max_age"),    # 8 hours
            httponly=server_conf.get("cookie_httponly"),     
            samesite=server_conf.get("cookie_samesite"),     
            secure  =server_conf.get("cookie_secure"),
            path    ="/"
        )
        logger.info(f"login success: {user_info['name']} (session: {session_id[:8]}...)")
        return {
            "message"       : "login success", 
            "user_id"       : user_info['user_id'],
            "name"          : user_info['name'],
            "department"    : user_info['department'],
            "position"      : user_info['position'],
            "security_level": user_info['security_level']
        }
    except (NotFoundError, UnauthorizedError) as e:
        logger.info(f"login failed: {e.message}")
        raise e


@sso_router.get("/session-info")
def get_session_info(coreiq_session: str = Cookie(None)):
    """현재 세션 정보 확인"""
    from repository.session import get_session_from_db
    if not coreiq_session :
        raise SessionNotFound("session not found")
    session_data = get_session_from_db(coreiq_session)          # 세션 데이터 가져오기
    if not session_data:
        raise SessionNotFound("session not found or expired")
    return {
        "user_id": session_data['user_id'],
        "username": session_data['username'],
        "name": session_data['name'],
        "security_level": session_data['security_level'],
        "expires_at":session_data['expires_at']
    }

@sso_router.post("/logout")
def logout(response: Response, coreiq_session: str = Cookie(None)):
    """로그아웃"""
    from repository.session import delete_session_from_db
    if coreiq_session:
        deleted = delete_session_from_db(coreiq_session)
        logger.info(f"logout deleted: session_id={coreiq_session}, deleted={deleted}")
    else:
        logger.info("logout called without session_id in cookie")
    response.delete_cookie("coreiq_session", path="/")
    return {"message": "logout success"}

@sso_router.get("/active-sessions")
def list_active_sessions():
    """활성 세션 목록 (관리용)"""
    from repository.session import list_all_sessions_from_db, delete_session_from_db
    sessions = list_all_sessions_from_db()
    current_kst = now_kst()
    active = []
    expired = []
    for s in sessions:
        if s["expires_at"] > current_kst:
            active.append(s)
        else:
            expired.append(s)
            delete_session_from_db(s["session_id"])  # 만료된 세션 삭제

    return {
        "active_sessions": len(active),
        "expired_cleaned": len(expired),
        "sessions": [
            {
                "user_id": s["user_id"],
                "name": s["name"],
                "created_at": s["created_at"],
                "expires_at": s["expires_at"]
            }
            for s in active
        ]
    }