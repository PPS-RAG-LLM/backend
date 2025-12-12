from __future__ import annotations
from typing import List, Optional, Tuple
import hashlib
from datetime import datetime

# Repository import
from repository.user import (
    UserQuery,
    find_users,
    find_user_by_id,
    create_user_record,
    update_user_record,
    bulk_update_users_records,
)

from storage.db_models import User
from utils.time import now_kst, to_kst_string
from sqlalchemy.exc import IntegrityError
from errors.exceptions import ConflictError

def _hash_password(raw: str) -> str:
    if not raw:
        return raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def list_users(qry: UserQuery) -> Tuple[List[User], int]:
    return find_users(qry)

def get_user(user_id: int) -> Optional[User]:
    return find_user_by_id(user_id)

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
    # 비즈니스 로직: 비밀번호 해싱
    hashed_pw = _hash_password(password) if password and len(password) < 50 else password
    
    # DB 저장 위임
    try:
        return create_user_record(
            username=username,
            name=name,
            password=hashed_pw,
            department=department,
            position=position,
            role=role or "user",
            security_level=security_level,
        )
    except IntegrityError:
        raise ConflictError("이미 존재하는 사용자 아이디입니다.")
    except Exception as e:
        # 기타 에러 처리
        if "integrity" in str(e).lower():
             raise ConflictError("이미 존재하는 사용자 아이디입니다.")
        raise e

def update_user(
    user_id: int,
    *,
    password: Optional[str] = None,
    expires_at: Optional[datetime] = None,
    **kwargs,
) -> User:

    update_data = {k: v for k, v in kwargs.items() if v is not None}
    
    # 비밀번호 해싱 처리
    if password is not None and password != "":
        update_data["password"] = _hash_password(password) if len(password) < 50 else password
    # 퇴사일 처리 로직
    if expires_at is not None or expires_at is None:
        # None이 들어오면 복구, 날짜가 들어오면 퇴사
        # (주의: expires_at이 키워드 인자로 들어왔는지 확인 필요)
        # 여기서는 expires_at 값을 명시적으로 처리
        # utils.time 함수 활용은 상황에 맞게 (Repository 내부에서 처리할 수도 있지만, 외부에서 받은 날짜 변환이 필요할 수 있음)
        pass 

    # 로직이 조금 복잡하므로 Repository update_user_record 호출 전 데이터 정리
    if expires_at is not None:
         # 타임존 제거 등 처리 후 전달
         if expires_at.tzinfo:
             expires_at = expires_at.replace(tzinfo=None)
         update_data["expires_at"] = expires_at
         update_data["suspended"] = 1
    elif "expires_at" in kwargs and kwargs["expires_at"] is None:
        # 명시적으로 None을 보낸 경우 (복구)
        update_data["expires_at"] = None
        update_data["suspended"] = 0

    u = update_user_record(user_id, **update_data)
    if not u:
        raise ValueError("사용자를 찾을 수 없습니다.")
    return u

       

def soft_delete_user(user_id: int, *, retired_at: Optional[datetime] = None) -> User:
    retired_at = retired_at or now_kst()
    if retired_at.tzinfo:
        retired_at = retired_at.replace(tzinfo=None)
    return update_user(user_id, expires_at=retired_at)

def restore_user(user_id: int) -> User:
    # 복구 로직: update_user_record 직접 호출해도 됨
    u = update_user_record(user_id, expires_at=None, suspended=0)
    if not u:
        raise ValueError("사용자를 찾을 수 없습니다.")
    return u

def bulk_update_users(ids: List[int], **kwargs) -> int:
    # None 값 제거
    data = {k: v for k, v in kwargs.items() if v is not None}
    return bulk_update_users_records(ids, **data)


def bulk_soft_delete(ids: List[int], *, retired_at: Optional[datetime] = None) -> int:
    retired_at = retired_at or now_kst()
    if retired_at.tzinfo:
        retired_at = retired_at.replace(tzinfo=None)
    
    return bulk_update_users_records(ids, expires_at=retired_at, suspended=1)
