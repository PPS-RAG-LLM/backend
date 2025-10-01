# routers/admin/manage_user_api.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal

from fastapi import APIRouter, Body, HTTPException, Query, Path
from pydantic import BaseModel, Field

from service.admin.manage_user import (
    UserQuery,
    list_users,
    get_user,
    create_user,
    update_user,
    soft_delete_user,
    restore_user,
    bulk_update_users,
    bulk_soft_delete,
)

router = APIRouter(
    prefix="/v1/admin/users",
    tags=["Admin Users"],
    responses={200: {"description": "Success"}},
)

# ---------- Schemas ----------
class UserOut(BaseModel):
    id: int
    role: str
    username: str
    name: str
    department: str
    position: str
    security_level: int
    suspended: int
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserCreateIn(BaseModel):
    role: Literal["user", "admin"] = "user"
    username: str = Field(..., min_length=3)
    name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=4, description="평문 또는 이미 해시된 값")
    department: str
    position: str
    security_level: int = 3
    # ⛔️ 제거: daily_message_limit / suspended / pfp_filename / bio


class UserUpdateIn(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    position: Optional[str] = None
    password: Optional[str] = Field(None, description="변경 시에만 전달(평문 가능)")
    role: Optional[Literal["user", "admin"]] = None
    security_level: Optional[int] = None
    # ⛔️ 제거: daily_message_limit / suspended / pfp_filename / bio
    expires_at: Optional[datetime] = Field(
        default=None, description="퇴사일 설정. NULL로 보내면 복구"
    )


class BulkUpdateIn(BaseModel):
    ids: List[int] = Field(..., min_items=1)
    department: Optional[str] = None
    position: Optional[str] = None
    role: Optional[Literal["user", "admin"]] = None
    security_level: Optional[int] = None
    # ⛔️ 제거: daily_message_limit / suspended


class BulkDeleteIn(BaseModel):
    ids: List[int] = Field(..., min_items=1)
    retired_at: Optional[datetime] = None


class ListResponse(BaseModel):
    items: List[UserOut]
    total: int
    page: int
    page_size: int


# ---------- Endpoints ----------
@router.get(
    "/list",
    response_model=ListResponse,
    summary="사용자 목록 조회",
    description=(
        "관리자 화면의 사용자 테이블 데이터를 조회합니다.\n\n"
        "### 기능\n"
        "- 부서/직급/검색어(q) 필터\n"
        "- 재직자만 보기(`active_only=True` → `expires_at IS NULL`)\n"
        "- 페이지네이션(page, page_size) 및 정렬(sort)\n\n"
        "### 정렬 키\n"
        "`created_at`, `name`, `username`, `department`, `position`, `security_level`, `expires_at`, `id`\n"
        "- 내림차순: `-created_at` (기본)\n"
        "- 오름차순: `name`\n\n"
        "### 비고\n"
        "- `q`는 `username`/`name`에 부분 일치로 검색합니다.\n"
        "- `expires_at`가 NULL이면 재직, 값이 있으면 퇴사일로 표시하세요."
    ),
)
def api_list_users(
    department: Optional[str] = Query(None, description="부서명 정확일치 필터"),
    position: Optional[str] = Query(None, description="직급 정확일치 필터"),
    q: Optional[str] = Query(None, description="아이디(username) 또는 이름(name) 부분 검색"),
    active_only: bool = Query(False, description="재직자만 보기(True면 퇴사자 제외)"),
    page: int = Query(1, ge=1, description="페이지(1부터)"),
    page_size: int = Query(20, ge=1, le=200, description="페이지 크기(1~200)"),
    sort: str = Query("-created_at", description="예: -created_at, name, security_level"),
):
    qry = UserQuery(
        department=department,
        position=position,
        q=q,
        active_only=active_only,
        page=page,
        page_size=page_size,
        sort=sort,
    )
    rows, total = list_users(qry)
    return ListResponse(
        items=[UserOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{user_id}",
    response_model=UserOut,
    summary="사용자 단건 조회",
    description="- `expires_at`가 NULL이면 재직, 값이 있으면 퇴사 처리된 사용자입니다.",
    responses={404: {"description": "사용자를 찾을 수 없음"}},
)
def api_get_user(user_id: int = Path(..., description="대상 사용자 ID")):
    u = get_user(user_id)
    if not u:
        raise HTTPException(404, "사용자를 찾을 수 없습니다.")
    return UserOut.model_validate(u)


@router.post(
    "",
    response_model=UserOut,
    status_code=201,
    summary="사용자 생성",
    description=(
        "신규 사용자를 생성합니다.\n"
        "- `username`은 유니크\n"
        "- `password`가 짧은 평문(길이<50)이면 서비스에서 **SHA-256 해시** 저장\n"
        "- 기본 상태: 재직(`expires_at` = NULL, `suspended` = 0)"
    ),
    responses={400: {"description": "유효성 오류 또는 username 중복"}},
)
def api_create_user(payload: UserCreateIn):
    try:
        u = create_user(**payload.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return UserOut.model_validate(u)



@router.patch("/{user_id}", response_model=UserOut)
def api_update_user(
    user_id: int = Path(..., description="대상 사용자 ID"),
    payload: UserUpdateIn = Body(..., description="부분 업데이트 페이로드(JSON)"),
):
    try:
        data = payload.model_dump(exclude_none=True)  # ✅ None 값 제거
        u = update_user(user_id, **data)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return UserOut.model_validate(u)

@router.post("/bulk/update")
def api_bulk_update(payload: BulkUpdateIn = Body(..., description="일괄 변경 페이로드(JSON)")):
    n = bulk_update_users(**payload.model_dump(exclude_none=True))  # ✅ None 값 제거
    return {"updated": n}


@router.delete(
    "/{user_id}",
    response_model=UserOut,
    summary="사용자 퇴사 처리(소프트 삭제)",
    description=(
        "실제 삭제 대신 퇴사일(`expires_at`)을 기록합니다. 서비스 레이어에서 `suspended=1` 자동 처리.\n"
        "- `retired_at` 미지정 시 서버 현재시각(KST)"
    ),
    responses={404: {"description": "사용자를 찾을 수 없음"}},
)
def api_soft_delete_user(
    user_id: int = Path(..., description="퇴사 처리할 사용자 ID"),
    retired_at: Optional[datetime] = Query(None, description="퇴사일(미지정 시 서버 현재시각)"),
):
    try:
        u = soft_delete_user(user_id, retired_at=retired_at)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return UserOut.model_validate(u)


@router.post(
    "/{user_id}/restore",
    response_model=UserOut,
    summary="사용자 복구",
    description="`expires_at`를 NULL로, `suspended=0`으로 되돌립니다.",
    responses={404: {"description": "사용자를 찾을 수 없음"}},
)
def api_restore_user(user_id: int = Path(..., description="복구할 사용자 ID")):
    try:
        u = restore_user(user_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return UserOut.model_validate(u)


@router.post(
    "/bulk/update",
    summary="사용자 일괄 변경",
    description="여러 사용자에 대해 `department`, `position`, `role`, `security_level`을 일괄 업데이트.",
)
def api_bulk_update(payload: BulkUpdateIn = Body(..., description="일괄 변경 페이로드(JSON)")):
    n = bulk_update_users(**payload.model_dump())
    return {"updated": n}


@router.post(
    "/bulk/delete",
    summary="사용자 일괄 퇴사 처리(소프트 삭제)",
    description="여러 사용자를 한 번에 퇴사 처리합니다. `retired_at` 미지정 시 서버 현재시각(KST).",
)
def api_bulk_delete(payload: BulkDeleteIn = Body(..., description="일괄 퇴사 처리 페이로드(JSON)")):
    n = bulk_soft_delete(payload.ids, retired_at=payload.retired_at)
    return {"deleted": n}
