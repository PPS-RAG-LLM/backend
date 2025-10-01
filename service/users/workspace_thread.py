from typing import Dict, Any
from utils import logger
from errors import BadRequestError, NotFoundError
from repository.workspace import get_workspace_id_by_slug_for_user
from repository.workspace_thread import (
    get_thread_by_slug_for_user,
    update_thread_name_by_slug_for_user,
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