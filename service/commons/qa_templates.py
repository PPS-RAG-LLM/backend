from repository.prompt_templates.qa_templates import (
    repo_create_qa_template, repo_list_qa_templates, repo_update_qa_template, repo_delete_qa_template
)
from typing import List, Dict, Optional

def list_qa_templates_all() -> List[Dict[str, str]]:
    return repo_list_qa_templates()


def generate_new_qa_prompt(system_prompt: str, user_prompt:Optional[str]) -> Dict[str,str]:
    return repo_create_qa_template(system_prompt, user_prompt)

def update_qa_template(template_id: int, system_prompt: str, user_prompt: Optional[str] = "") -> Optional[Dict[str, str]]:
    return repo_update_qa_template(template_id, system_prompt, user_prompt)

def delete_qa_template(template_id: int) -> bool:
    return repo_delete_qa_template(template_id)
