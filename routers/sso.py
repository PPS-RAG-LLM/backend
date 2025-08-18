# routers/sso.py (완전 버전)
from fastapi import APIRouter, HTTPException, Response, Cookie, Depends
from pydantic import BaseModel
from utils import get_db, now_kst
import secrets, hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from errors import SessionNotFound, NotFoundError, UnauthorizedError
from service.users.session import create_session
from utils import logger

logger = logger(__name__)

sso_router = APIRouter(prefix="/v1/sso", tags=["SSO"])

# 메모리 세션 스토어 (실제로는 Redis 사용 권장)
active_sessions: Dict[str, Dict[str, Any]] = {}

class CompanyUserInfo(BaseModel):
    username: str      # 회사의 employee_id
    password: str

@sso_router.post("/login")
def sso_login(company_user: CompanyUserInfo, response: Response):
    """회사에서 넘어온 ID/PW로 로그인 및 세션 생성"""
    
    try:
        # 1. 로그인 처리 및 세션 생성
        session_id, user_info = create_session(company_user.username, company_user.password)
        
        # 2. 브라우저에 쿠키 설정
        response.set_cookie(
            key="pps_session",
            value=session_id,
            max_age=8*60*60,  # 8시간
            httponly=True,
            samesite="lax",
            path="/"
        )
        
        logger.info(f"✅ 로그인 성공: {user_info['name']} (세션: {session_id[:8]}...)")
        
        return {
            "message": "로그인 성공", 
            "user_id": user_info['user_id'],
            "name": user_info['name'],
            "department": user_info['department'],
            "position": user_info['position'],
            "security_level": user_info['security_level']
        }
        
    except (NotFoundError, UnauthorizedError) as e:
        logger.info(f"❌ 로그인 실패: {e.message}")
        raise e


@sso_router.get("/session-info")
def get_session_info(pps_session: str = Cookie(None)):
    """현재 세션 정보 확인"""
    
    if not pps_session or pps_session not in active_sessions:
        raise SessionNotFound("세션이 없습니다")
    
    session_data = active_sessions[pps_session]
    
    if session_data['expires_at'] < now_kst():
        del active_sessions[pps_session]
        raise SessionNotFound("세션이 만료되었습니다")
    
    return {
        "user_id": session_data['user_id'],
        "username": session_data['username'],
        "name": session_data['name'],
        "security_level": session_data['security_level'],
        "expires_at": session_data['expires_at'].isoformat()
    }

@sso_router.post("/logout")
def logout(pps_session: str = Cookie(None), response: Response = None):
    """로그아웃"""
    
    if pps_session and pps_session in active_sessions:
        del active_sessions[pps_session]
    
    # 쿠키 삭제
    response.delete_cookie("pps_session", path="/")
    
    return {"message": "로그아웃 완료"}

@sso_router.get("/active-sessions")
def list_active_sessions():
    """활성 세션 목록 (관리용)"""
    
    current_time = datetime.now()
    active_count = 0
    expired_sessions = []
    
    for session_id, session_data in active_sessions.items():
        if session_data['expires_at'] > now_kst():
            active_count += 1
        else:
            expired_sessions.append(session_id)
    
    # 만료된 세션 정리
    for expired_id in expired_sessions:
        del active_sessions[expired_id]
    
    return {
        "active_sessions": active_count,
        "expired_cleaned": len(expired_sessions),
        "sessions": [
            {
                "user_id": data['user_id'],
                "name": data['name'],
                "created_at": data['created_at'].isoformat(),
                "expires_at": data['expires_at'].isoformat()
            }
            for data in active_sessions.values()
            if data['expires_at'] > now_kst()
        ]
    }