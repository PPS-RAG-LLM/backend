from repository.prompt_templates.qa_templates import repo_list_qa_templates
from typing import List, Dict

def list_qa_templates_all() -> List[Dict[str, str]]:
    """관리자 전체보기용"""
    return repo_list_qa_templates()