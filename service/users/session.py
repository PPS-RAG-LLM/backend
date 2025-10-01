from repository.session import create_new_session, get_session_from_db, delete_sessions_by_user_id
from repository.user import get_user_by_username
from errors import SessionNotFound, NotFoundError, UnauthorizedError
import hashlib
from utils import logger

logger = logger(__name__)

def create_session(username: str, password: str) -> tuple[str, dict]:
    """로그인 처리 및 세션 생성"""
    
    # 1. 사용자 조회
    user = get_user_by_username(username)
    if not user:
        raise NotFoundError("user not found")

    # 2. 비밀번호 검증
    db_hash = user["password"]
    input_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if db_hash != input_hash:
        raise UnauthorizedError("wrong password")
    logger.info(f"hashed password(SHA256) verified: {user['name']}")

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

