from typing import List, Dict, Optional
from repository.prompt_templates.summary_templates import (
    repo_list_summary_templates,
    repo_get_summary_template_by_id,
)

def list_summary_templates() -> List[Dict[str, str]]:
    """사용자 디폴트 프롬프트"""
    return repo_list_summary_templates(default_only=True)

def list_summary_templates_all() -> List[Dict[str, str]]:
    """관리자 전체보기용"""
    return repo_list_summary_templates(default_only=False)

def get_summary_template(template_id: int) -> Optional[Dict[str, str]]:
    return repo_get_summary_template_by_id(template_id)