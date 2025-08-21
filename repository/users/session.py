from utils import get_db, now_kst_string, expires_at_kst
import json
from datetime import datetime, timedelta
import secrets
from utils import logger

logger = logger(__name__)

def create_new_session(user_id: int) -> str:
    """새 세션 생성 및 저장"""
    session_id = secrets.token_urlsafe(32)
    save_session_to_db(session_id, user_id)
    return session_id

# 메모리 세션 대신 DB 세션 사용
def save_session_to_db(session_id: str, user_id: dict):
    """세션을 DB에 저장"""
    db = get_db()
    try:

        db.execute(
            """INSERT OR REPLACE INTO user_sessions 
               (session_id, user_id, created_at, expires_at) 
               VALUES (?, ?, ?, ?)""",
            (
                session_id,
                user_id, 
                now_kst_string(),
                expires_at_kst()
            )
        )
        db.commit()

    finally:
        db.close()

def get_session_from_db(session_id: str) -> dict:
    """세션 조회 (users 테이블과 JOIN해서 사용자 정보도 함께)"""
    db = get_db()
    try:
        current_kst = now_kst_string()
        
        result = db.execute(
            """SELECT u.id, u.username, u.name, u.department, u.position, u.security_level, s.expires_at
               FROM user_sessions s 
               JOIN users u ON s.user_id = u.id
               WHERE s.session_id = ?""",
            (session_id,)
        ).fetchone()
        
        if not result:
            return None
            
        if result[6] <= current_kst:  # expires_at 체크
            db.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
            db.commit()
            logger.info(f"delete expired session: {session_id}")
            return None
        
        return {
            'user_id': result[0],
            'username': result[1], 
            'name': result[2],
            'department': result[3],
            'position': result[4],
            'security_level': result[5],
            'expires_at': result[6]
        }
    finally:
        db.close()

def list_all_sessions_from_db() -> list[dict]:
    """DB에 저장된 모든 세션 정보를 조회합니다."""
    db = get_db()
    try:
        results = db.execute(
            """SELECT s.session_id, u.id, u.username, u.name, u.department, u.position, u.security_level, s.created_at, s.expires_at
               FROM user_sessions s
               JOIN users u ON s.user_id = u.id"""
        ).fetchall()
        return [
            {
                "session_id": row[0],
                "user_id": row[1],
                "username": row[2],
                "name": row[3],
                "department": row[4],
                "position": row[5],
                "security_level": row[6],
                "created_at": row[7],
                "expires_at": row[8],
            }
            for row in results
        ]
    finally:
        db.close()

def delete_session_from_db(session_id: str):
    """DB에서 세션 삭제"""
    db = get_db()
    try:
        db.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
        db.commit()
        # 삭제된 세션이 있는지 확인
        remaining = db.execute("SELECT 1 FROM user_sessions WHERE session_id = ?", (session_id,)).fetchone()
        return remaining is None
    finally:
        db.close()
        

def delete_sessions_by_user_id(user_id: int) :
    """사용자 ID에 해당하는 모든 세션 삭제"""
    db= get_db()
    try:
        db.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        db.commit()
        logger.info(f"delete sessions by user_id: {user_id}")
    finally:
        db.close()


def cleanup_expired_sessions():
    """만료된 모든 세션 일괄 정리"""
    db = get_db()
    try:
        current_kst = now_kst_string()
        
        # 만료된 세션 개수 확인
        count_result = db.execute(
            "SELECT COUNT(*) FROM user_sessions WHERE expires_at <= ?",
            (current_kst,)
        ).fetchone()
        
        expired_count = count_result[0] if count_result else 0
        
        # 만료된 세션 삭제
        db.execute(
            "DELETE FROM user_sessions WHERE expires_at <= ?", 
            (current_kst,)
        )
        db.commit()
        
        if expired_count > 0:
            logger.info(f"clean up expired sessions: {expired_count} sessions")
            
        return expired_count
    finally:
        db.close()