# /home/work/CoreIQ/yb/backend/service/users/doc_gen_templates.py
from typing import List, Dict, Optional
from repository.users.doc_gen_templates import (
    repo_list_doc_gen_templates,
    repo_get_doc_gen_template_by_id_with_vars,
)

def list_doc_gen_templates() -> List[Dict[str, str]]:
    return repo_list_doc_gen_templates(default_only=True)
    
def list_doc_gen_templates_all() -> List[Dict[str, str]]:
    return repo_list_doc_gen_templates(default_only=False)

def get_doc_gen_template(template_id: int) -> Optional[Dict[str, object]]:
    return repo_get_doc_gen_template_by_id_with_vars(template_id)