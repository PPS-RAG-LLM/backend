from typing import List, Dict, Optional
from errors import NotFoundError
from repository.users.workspace import get_workspace_id_by_slug_for_user
from repository.users.summary_templates import (
    repo_list_summary_templates,
    repo_get_summary_template_by_id,
)

def list_summary_templates() -> List[Dict[str, str]]:
    return repo_list_summary_templates()

def get_summary_template(template_id: int) -> Optional[Dict[str, str]]:
    return repo_get_summary_template_by_id(template_id)