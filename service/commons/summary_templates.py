from typing import List, Dict, Optional
from repository.prompt_templates.summary_templates import (
    repo_create_summary_template,
    repo_list_summary_templates,
    repo_get_summary_template_by_id,
    repo_update_summary_template,
    repo_delete_summary_template
)

def list_summary_templates() -> List[Dict[str, str]]:
    """사용자 디폴트 프롬프트"""
    return repo_list_summary_templates(default_only=True)

def list_summary_templates_all() -> List[Dict[str, str]]:
    """관리자 전체보기용"""
    return repo_list_summary_templates(default_only=False)

def get_summary_template(template_id: int) -> Optional[Dict[str, str]]:
    return repo_get_summary_template_by_id(template_id)

def generate_summary_template(system_prompt:str, user_prompt: Optional[str]=""):
    return repo_create_summary_template(system_prompt, user_prompt)

def update_summary_template(template_id: int, system_prompt: str, user_prompt: Optional[str] = "") -> Optional[Dict[str, str]]:
    return repo_update_summary_template(template_id, system_prompt, user_prompt)

def delete_summary_template(template_id: int) -> bool:
    return repo_delete_summary_template(template_id)
