from typing import Dict, Any, List, Optional, Tuple
from utils import now_kst
from utils.crypto import decrypt_data, encrypt_data
from utils.database import get_session
from sqlalchemy import select
from storage.db_models import User
from sqlalchemy import select, func, or_, update
from sqlalchemy.orm import Session


class UserQuery:
    def __init__(
        self,
        department: Optional[str] = None,
        position: Optional[str] = None,
        q: Optional[str] = None,
        active_only: bool = False,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ):
        self.department = department
        self.position = position
        self.q = q
        self.active_only = active_only
        self.page = max(1, page)
        self.page_size = max(1, min(200, page_size))
        self.sort = sort

    def apply(self, stmt, with_sort: bool = True):
        if self.department:
            stmt = stmt.where(User.department == self.department)
        if self.position:
            stmt = stmt.where(User.position == self.position)
        if self.q:
            like = f"%{self.q}%"
            stmt = stmt.where(or_(User.username.ilike(like), User.name.ilike(like)))
        if self.active_only:
            stmt = stmt.where(User.expires_at.is_(None))
        
        if with_sort:
            key = self.sort.lstrip("+-")
            desc = self.sort.startswith("-")
            sort_col = {
                "created_at": User.created_at,
                "name": User.name,
                "username": User.username,
                "department": User.department,
                "position": User.position,
                "security_level": User.security_level,
                "expires_at": User.expires_at,
                "id": User.id,
            }.get(key, User.created_at)
            stmt = stmt.order_by(sort_col.desc() if desc else sort_col.asc())
        return stmt

    def paginate(self, stmt):
        offset = (self.page - 1) * self.page_size
        return stmt.offset(offset).limit(self.page_size)


def find_users(qry: UserQuery) -> Tuple[List[User], int]:
    with get_session() as s:
        base = select(User)
        base = qry.apply(base, with_sort=True)
        
        count_stmt = select(func.count()).select_from(User)
        count_stmt = qry.apply(count_stmt, with_sort=False)
        
        total = s.execute(count_stmt).scalar_one()
        rows = s.execute(qry.paginate(base)).scalars().all()
        return rows, total

def find_user_by_id(user_id: int) -> Optional[User]:
    with get_session() as s:
        return s.get(User, user_id)

def create_user_record(
    username: str,
    name: str,
    password: str,
    department: str,
    position: str,
    role: str = "user",
    security_level: int = 3,
) -> User:
    now = now_kst()
    item = User(
        username=username,
        name=name,
        password=password,
        department=department,
        position=position,
        role=role,
        security_level=security_level,
        suspended=0,
        pfp_filename=None,
        bio="",
        created_at=now,
        updated_at=now,
        expires_at=None,
    )
    with get_session() as s:
        s.add(item)
        s.commit()
        s.refresh(item)
        return item

def update_user_record(user_id: int, **kwargs) -> Optional[User]:
    with get_session() as s:
        u = s.get(User, user_id)
        if not u:
            return None
        
        for k, v in kwargs.items():
            if hasattr(u, k):
                setattr(u, k, v)
        
        u.updated_at = now_kst()
        s.commit()
        s.refresh(u)
        return u

def bulk_update_users_records(ids: List[int], **kwargs) -> int:
    if not ids:
        return 0
    with get_session() as s:
        # bulk update는 ORM 객체 로딩 없이 update 구문 실행
        stmt = update(User).where(User.id.in_(ids)).values(updated_at=now_kst(), **kwargs)
        result = s.execute(stmt)
        s.commit()
        return result.rowcount

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
                "id"            : m["id"],
                "role"          : m["role"],
                "username"      : m["username"],
                "name"          : m["name"],
                "department"    : m["department"],
                "position"      : m["position"],
                "security_level": m["security_level"],
            }
    finally:
        pass

def update_user_api_keys(user_id: int, updates: Dict[str, Any]):
    from utils.database import get_session
    from storage.db_models import User
    
    # [수정] 값 암호화 처리 후 업데이트
    encrypted_updates = {}
    for k, v in updates.items():
        if v and isinstance(v, str):
            encrypted_updates[k] = encrypt_data(v)
        else:
            encrypted_updates[k] = v

    session = get_session()
    try:
        session.query(User).filter(User.id == user_id).update(encrypted_updates)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_api_keys_by_user_id(user_id: int) -> Optional[Dict[str, Any]]:
    with get_session() as session:
        stmt = select(
            User.openai_api_key,
            User.anthropic_api_key,
            User.gemini_api_key,
        ).where(User.id == user_id).limit(1)
        row = session.execute(stmt).first()
        
        if not row:
            return None
            
        # [수정] 복호화하여 반환 (Row 객체를 딕셔너리로 변환)
        # decrypt_data는 평문인 경우 그대로 반환하므로 기존 데이터와 호환됩니다.
        return {
            "openai_api_key"    : decrypt_data(row.openai_api_key),
            "anthropic_api_key" : decrypt_data(row.anthropic_api_key),
            "gemini_api_key"    : decrypt_data(row.gemini_api_key),
        }