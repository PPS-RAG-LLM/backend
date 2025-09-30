# service/admin/manage_user.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo  # [ADD]

from sqlalchemy import or_, select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from utils.database import get_session
from storage.db_models import User

import hashlib

# --------- KST Helpers ---------
KST = ZoneInfo("Asia/Seoul")  # [ADD]

def _now_kst_naive() -> datetime:
    """tz 없는 KST(Asia/Seoul) naive datetime."""
    return datetime.now(KST).replace(tzinfo=None)  # [ADD]

def _to_kst_naive(dt: Optional[datetime]) -> Optional[datetime]:
    """전달된 datetime을 tz 상관없이 KST naive로 변환."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt  # 이미 naive → KST로 들어온 값으로 간주
    return dt.astimezone(KST).replace(tzinfo=None)  # [ADD]


# --------- Utils ---------
def _hash_password(raw: str) -> str:
    if not raw:
        return raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# --------- Query DTO ---------
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
        self.department = department or None
        self.position = position or None
        self.q = q or None
        self.active_only = active_only
        self.page = max(1, page)
        self.page_size = max(1, min(200, page_size))
        self.sort = sort

    def apply(self, stmt):
        if self.department:
            stmt = stmt.where(User.department == self.department)
        if self.position:
            stmt = stmt.where(User.position == self.position)
        if self.q:
            like = f"%{self.q}%"
            stmt = stmt.where(or_(User.username.ilike(like), User.name.ilike(like)))
        if self.active_only:
            stmt = stmt.where(User.expires_at.is_(None))
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


# --------- Core CRUD ---------
def list_users(qry: UserQuery) -> Tuple[List[User], int]:
    with get_session() as s:  # type: Session
        base = select(User)
        base = qry.apply(base)

        count_stmt = select(func.count()).select_from(User)
        count_stmt = qry.apply(count_stmt)

        total = s.execute(count_stmt).scalar_one()
        rows = s.execute(qry.paginate(base)).scalars().all()
        return rows, total


def get_user(user_id: int) -> Optional[User]:
    with get_session() as s:
        return s.get(User, user_id)


def create_user(
    *,
    role: str,
    username: str,
    name: str,
    password: str,
    department: str,
    position: str,
    daily_message_limit: Optional[int] = None,
    security_level: int = 3,
    suspended: int = 0,
    pfp_filename: Optional[str] = None,
    bio: str = "",
) -> User:
    now = _now_kst_naive()  # [CHG] UTC → KST
    item = User(
        role=role or "user",
        username=username,
        name=name,
        password=_hash_password(password) if password and len(password) < 50 else password,
        department=department,
        position=position,
        daily_message_limit=daily_message_limit,
        security_level=security_level,
        suspended=suspended,
        pfp_filename=pfp_filename,
        bio=bio or "",
        created_at=now,   # [CHG]
        updated_at=now,   # [CHG]
        expires_at=None,  # 신규는 재직
    )

    with get_session() as s:
        s.add(item)
        try:
            s.commit()
        except IntegrityError as e:
            s.rollback()
            raise ValueError("이미 존재하는 사용자(username)입니다.") from e
        s.refresh(item)
        return item


def update_user(
    user_id: int,
    *,
    name: Optional[str] = None,
    department: Optional[str] = None,
    position: Optional[str] = None,
    password: Optional[str] = None,
    role: Optional[str] = None,
    daily_message_limit: Optional[int] = None,
    suspended: Optional[int] = None,
    security_level: Optional[int] = None,
    pfp_filename: Optional[str] = None,
    bio: Optional[str] = None,
    expires_at: Optional[datetime] = None,
) -> User:
    with get_session() as s:
        u: User = s.get(User, user_id)
        if not u:
            raise ValueError("사용자를 찾을 수 없습니다.")

        if name is not None:
            u.name = name
        if department is not None:
            u.department = department
        if position is not None:
            u.position = position
        if password is not None and password != "":
            u.password = _hash_password(password) if len(password) < 50 else password
        if role is not None:
            u.role = role
        if daily_message_limit is not None:
            u.daily_message_limit = daily_message_limit
        if suspended is not None:
            u.suspended = int(bool(suspended))
        if security_level is not None:
            u.security_level = security_level
        if pfp_filename is not None:
            u.pfp_filename = pfp_filename
        if bio is not None:
            u.bio = bio
        if expires_at is not None or expires_at is None:
            u.expires_at = _to_kst_naive(expires_at)  # [CHG] 일관 KST 저장

        u.updated_at = _now_kst_naive()  # [CHG]
        s.commit()
        s.refresh(u)
        return u


def soft_delete_user(user_id: int, *, retired_at: Optional[datetime] = None) -> User:
    """실제 삭제 대신 퇴사일(expires_at)만 기록."""
    retired_at = _to_kst_naive(retired_at) or _now_kst_naive()  # [CHG]
    return update_user(user_id, expires_at=retired_at, suspended=1)


def restore_user(user_id: int) -> User:
    """퇴사 처리 복구 (expires_at NULL, suspended 0)."""
    return update_user(user_id, expires_at=None, suspended=0)


def bulk_update_users(
    ids: List[int],
    *,
    department: Optional[str] = None,
    position: Optional[str] = None,
    role: Optional[str] = None,
    security_level: Optional[int] = None,
    suspended: Optional[int] = None,
) -> int:
    if not ids:
        return 0
    with get_session() as s:
        q = s.query(User).filter(User.id.in_(ids))
        updated = 0
        now = _now_kst_naive()  # [CHG]
        for u in q:
            if department is not None:
                u.department = department
            if position is not None:
                u.position = position
            if role is not None:
                u.role = role
            if security_level is not None:
                u.security_level = security_level
            if suspended is not None:
                u.suspended = int(bool(suspended))
            u.updated_at = now
            updated += 1
        s.commit()
        return updated


def bulk_soft_delete(ids: List[int], *, retired_at: Optional[datetime] = None) -> int:
    if not ids:
        return 0
    retired_at = _to_kst_naive(retired_at) or _now_kst_naive()  # [CHG]
    with get_session() as s:
        q = s.query(User).filter(User.id.in_(ids))
        n = 0
        now = _now_kst_naive()  # [CHG]
        for u in q:
            u.expires_at = retired_at
            u.suspended = 1
            u.updated_at = now
            n += 1
        s.commit()
        return n
