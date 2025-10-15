from typing import Dict, Any
from errors.exceptions import DatabaseError
from utils import logger, generate_thread_slug
from errors import BadRequestError, NotFoundError
from repository.workspace import get_workspace_id_by_slug_for_user
from repository.workspace_thread import (
    get_thread_by_slug_for_user,
    update_thread_name_by_slug_for_user,
    create_default_thread
)

log = logger(__name__)

def update_thread_name_for_workspace(
    user_id: int, slug: str, thread_slug: str, name: str
) -> Dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise BadRequestError("name은 비어 있을 수 없습니다")

    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("Workspace를 찾을 수 없습니다")

    thread = get_thread_by_slug_for_user(user_id, thread_slug)
    if not thread or thread.get("workspace_id") != workspace_id:
        raise NotFoundError("Thread를 찾을 수 없습니다")

    updated = update_thread_name_by_slug_for_user(user_id, thread_slug, name.strip())
    if not updated:
        raise NotFoundError("Thread 업데이트에 실패했습니다")

    return {
        "thread": {
            "id": updated["id"],
            "name": updated["name"],
            "slug": updated["slug"],
            "user_id": updated["user_id"],
            "workspace_id": updated["workspace_id"],
        },
        "message": None,
    }

def create_new_workspace_thread_for_workspace(user_id: int, slug: str, name: str) -> Dict[str, Any]:
    workspace_id = get_workspace_id_by_slug_for_user(user_id, slug)
    if not workspace_id:
        raise NotFoundError("Workspace를 찾을 수 없습니다")

    thread_slug = generate_thread_slug(name) # thread_slug 생성
    thread_id = create_default_thread(user_id=user_id, name=name, thread_slug=thread_slug, workspace_id=workspace_id)
    
    # 생성된 스레드 정보를 조회
    from repository.workspace_thread import get_thread_by_id
    thread = get_thread_by_id(thread_id)
    
    if not thread:
        raise DatabaseError("Thread 생성 후 조회에 실패했습니다")
    
    return {
        "thread": {
            "id": thread["id"],
            "name": thread["name"],
            "threadSlug": thread["slug"],
            "workspaceId": thread["workspace_id"],
        },
        "message": "새로운 스레드가 생성되었습니다",
    }