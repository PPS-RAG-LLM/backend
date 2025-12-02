from typing import Optional, Dict, Any
from utils.database import get_session
from sqlalchemy import select
from storage.db_models import User


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    try:
        with get_session() as session:
            stmt = select(
                User.id,
                User.role,
                User.username,
                User.name,
                User.department,
                User.position,
                User.security_level,
            ).where(User.username == username).limit(1)
            row = session.execute(stmt).first()
            if not row:
                return None
            # 호출부 호환을 위해 dict 형태로 반환
            m = row._mapping
            return {
                "id": m["id"],
                "role": m["role"],
                "username": m["username"],
                "name": m["name"],
                "department": m["department"],
                "position": m["position"],
                "security_level": m["security_level"],
            }
    finally:
        pass

