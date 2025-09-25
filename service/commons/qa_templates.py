from repository.prompt_templates.qa_templates import repo_create_qa_template, repo_list_qa_templates
from typing import List, Dict, Optional

def list_qa_templates_all() -> List[Dict[str, str]]:
    """관리자 전체보기용"""
    return repo_list_qa_templates()


def generate_new_qa_prompt(system_prompt: str, user_prompt:Optional[str]) -> Dict[str,str]:
    return repo_create_qa_template(system_prompt, user_prompt)