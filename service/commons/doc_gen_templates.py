# /home/work/CoreIQ/yb/backend/service/users/doc_gen_templates.py
from typing import List, Dict, Optional
from repository.prompt_templates.doc_gen_templates import (
    repo_list_doc_gen_templates,
    repo_get_doc_gen_template_by_id_with_vars,
    repo_create_doc_gen_template,
    repo_update_doc_gen_template,
    repo_delete_doc_gen_template,
    repo_delete_doc_gen_prompt_variable
)

def list_doc_gen_templates() -> List[Dict[str, str]]:
    return repo_list_doc_gen_templates(default_only=True)
    
def list_doc_gen_templates_all() -> List[Dict[str, str]]:
    return repo_list_doc_gen_templates(default_only=False)

def get_doc_gen_template(template_id: int) -> Optional[Dict[str, object]]:
    return repo_get_doc_gen_template_by_id_with_vars(template_id)

def generate_new_doc_gen_prompt(name, system_prompt, user_prompt, variables) ->  Optional[Dict[str, object]]:
    return repo_create_doc_gen_template(name, system_prompt, user_prompt, variables)

def update_doc_gen_prompt_service(
    template_id: int, name: str, system_prompt: str, user_prompt: Optional[str], variables: Optional[List[Dict[str, object]]]
    ) -> Optional[Dict[str, object]]:
    return repo_update_doc_gen_template(template_id, name, system_prompt, user_prompt, variables)

def remove_doc_gen_template(template_id: int) -> bool:
    return repo_delete_doc_gen_template(template_id)

def remove_doc_gen_prompt_variable(template_id: int, variable_id: int) -> bool:
    return repo_delete_doc_gen_prompt_variable(template_id, variable_id)