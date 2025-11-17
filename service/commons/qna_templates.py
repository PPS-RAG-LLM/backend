from repository.prompt_templates.qna_templates import (
    repo_create_qna_template, repo_list_qna_templates, repo_update_qna_template, repo_delete_qna_template
)
from typing import List, Dict, Optional

def list_qna_templates_all() -> List[Dict[str, str]]:
    return repo_list_qna_templates()


def generate_new_qna_prompt(system_prompt: str, user_prompt:Optional[str]) -> Dict[str,str]:
    return repo_create_qna_template(system_prompt, user_prompt)

def update_qna_template(template_id: int, system_prompt: str, user_prompt: Optional[str] = "") -> Optional[Dict[str, str]]:
    return repo_update_qna_template(template_id, system_prompt, user_prompt)

def delete_qna_template(template_id: int) -> bool:
    return repo_delete_qna_template(template_id)
