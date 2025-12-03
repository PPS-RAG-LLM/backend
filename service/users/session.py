import bcrypt
from config import config
from repository.session import create_new_session, delete_sessions_by_user_id
from repository.user import get_user_by_username
from errors import SessionNotFound, NotFoundError, UnauthorizedError
import hashlib
from utils import logger
import jwt  # PyJWT

logger = logger(__name__)

def create_session(username: str, password: str) -> tuple[str, dict]:
    """로그인 처리 및 세션 생성"""
    
    # 1. 사용자 조회
    user = get_user_by_username(username)
    if not user:
        raise NotFoundError("user not found")

    # 2. 비밀번호 검증
    if not user["password"]:
        raise UnauthorizedError("SSO only user")

    db_hash = user["password"].encode("utf-8") if isinstance(user["password"], str) else user["password"]

    try:
        # bcrypt로 비밀번호 검증
        if not bcrypt.checkpw(password.encode("utf-8"), db_hash):
            raise UnauthorizedError("wrong password")
    except ValueError:
        # 기존 SHA256 해시와 호환이 안 되므로 초기화 필요 등의 로그 남김
        raise UnauthorizedError("password format error")

    logger.info(f"password verified: {user['name']}")

    # 이전 세션 삭제
    delete_sessions_by_user_id(user['id'])

    # 3. 세션 생성 및 저장
    session_id = create_new_session(user['id'])
    
    # 4. 사용자 정보 반환 (password 제외)
    user_info = {
        'user_id': user['id'],
        'username': user['username'],
        'name': user['name'],
        'department': user['department'],
        'position': user['position'],
        'security_level': user['security_level']
    }
    return session_id, user_info


def login_by_sso_token(token: str) -> tuple[str, dict]:
    """
    SSO용 신뢰된 토큰 검증 및 로그인 처리
    비밀번호 검사 없음. 토큰 서명이 맞으면 로그인 허용.
    """
    secret_key = config.get("server").get("sso_secret_key")
    try:
        # 1. 토큰 서명 검증 (HS256)
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        username = payload.get("username")
        if not username:
            raise UnauthorizedError("invalid token payload")
            
        logger.info(f"SSO Token verified for user: {username}")
    except jwt.ExpiredSignatureError:
        raise UnauthorizedError("token expired")
    except jwt.InvalidTokenError:
        raise UnauthorizedError("invalid token")

    # 2. 사용자 조회 (DB에 사용자가 존재해야 함 - 미들웨어 동기화 전제)
    user = get_user_by_username(username)
    if not user:
        # 정책 결정 필요: 없을 때 자동 생성할지, 에러를 낼지.
        # 여기서는 에러 처리 (미들웨어가 먼저 데이터를 넣어야 함)
        raise NotFoundError(f"User {username} not synced yet")

    # 3. 세션 생성 (비밀번호 확인 생략)
    # SSO 로그인은 기존 세션을 끊지 않거나, 끊거나 정책 결정. 여기선 동일하게 처리.
    delete_sessions_by_user_id(user['id'])
    session_id = create_new_session(user['id'])

    user_info = {
        'user_id': user['id'],
        'role': user['role'],
        'username': user['username'],
        'name': user['name'],
        'department': user['department'],
        'position': user['position'],
        'security_level': user['security_level']
    }
    return session_id, user_info