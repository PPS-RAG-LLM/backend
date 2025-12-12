# service/admin/manage_user.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select, func, update  # ← update 추가
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from utils.database import get_session
from storage.db_models import User

import hashlib

KST = ZoneInfo("Asia/Seoul")

def _now_kst_naive() -> datetime:
    return datetime.now(KST).replace(tzinfo=None)

def _to_kst_naive(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(KST).replace(tzinfo=None)

def _hash_password(raw: str) -> str:
    if not raw:
        return raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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

def list_users(qry: UserQuery) -> Tuple[List[User], int]:
    with get_session() as s:
        base = select(User)
        base = qry.apply(base, with_sort=True)
        count_stmt = select(func.count()).select_from(User)
        count_stmt = qry.apply(count_stmt, with_sort=False)
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
    security_level: int = 3,
) -> User:
    now = _now_kst_naive()
    item = User(
        role=role or "user",
        username=username,
        name=name,
        password=_hash_password(password) if password and len(password) < 50 else password,
        department=department,
        position=position,
        security_level=security_level,
        suspended=0,
        pfp_filename=None,
        bio="",
        created_at=now,
        updated_at=now,
        expires_at=None,  # ✅ 재직 상태로 명시
    )
    with get_session() as s:
        s.add(item)
        s.flush()  # INSERT 실행

        # ✅ 추가 안전장치: DB server_default로 값이 들어갔다면 NULL로 되돌림
        try:
            s.refresh(item)
            if item.expires_at is not None:
                s.execute(
                    update(User).where(User.id == item.id).values(expires_at=None, suspended=0)
                )
        except Exception:
            # 실패해도 아래 commit에서 적어도 재직 로직은 유지
            pass

        s.commit()
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
    security_level: Optional[int] = None,
    expires_at: Optional[datetime] = None,
    **_ignore,  # ✅ 모르는 키 무시
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
        if security_level is not None:
            u.security_level = security_level

        # 퇴사일 변경 (NULL 복구 포함)
        if expires_at is not None or expires_at is None:
            u.expires_at = _to_kst_naive(expires_at)
            # 퇴사/복구에 따른 suspended 자동 처리
            u.suspended = 1 if u.expires_at is not None else 0

        u.updated_at = _now_kst_naive()
        s.commit()
        s.refresh(u)
        return u

def soft_delete_user(user_id: int, *, retired_at: Optional[datetime] = None) -> User:
    retired_at = _to_kst_naive(retired_at) or _now_kst_naive()
    return update_user(user_id, expires_at=retired_at)

def restore_user(user_id: int) -> User:
    return update_user(user_id, expires_at=None)

def bulk_update_users(
    ids: List[int],
    *,
    department: Optional[str] = None,
    position: Optional[str] = None,
    role: Optional[str] = None,
    security_level: Optional[int] = None,
    **_ignore,  # ✅ 모르는 키 무시
) -> int:
    if not ids:
        return 0
    with get_session() as s:
        q = s.query(User).filter(User.id.in_(ids))
        updated = 0
        now = _now_kst_naive()
        for u in q:
            if department is not None:
                u.department = department
            if position is not None:
                u.position = position
            if role is not None:
                u.role = role
            if security_level is not None:
                u.security_level = security_level
            u.updated_at = now
            updated += 1
        s.commit()
        return updated

def bulk_soft_delete(ids: List[int], *, retired_at: Optional[datetime] = None) -> int:
    if not ids:
        return 0
    retired_at = _to_kst_naive(retired_at) or _now_kst_naive()
    with get_session() as s:
        q = s.query(User).filter(User.id.in_(ids))
        n = 0
        now = _now_kst_naive()
        for u in q:
            u.expires_at = retired_at
            u.suspended = 1
            u.updated_at = now
            n += 1
        s.commit()
        return n
