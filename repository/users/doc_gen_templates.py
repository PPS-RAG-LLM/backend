from typing import List, Dict, Optional
from sqlalchemy import select
from utils.database import get_session
from storage.db_models import SystemPromptTemplate, PromptMapping, SystemPromptVariable

def repo_list_doc_gen_templates() -> List[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(SystemPromptTemplate.id, SystemPromptTemplate.name)
            .where(
                SystemPromptTemplate.category == "doc_gen",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        rows = session.execute(stmt).all()
        return [{"id": r.id, "name": r.name} for r in rows]

def repo_get_doc_gen_template_by_id_with_vars(template_id: int) -> Optional[Dict[str, object]]:
    with get_session() as session:
        tmpl_stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.content,
            )
            .where(
                SystemPromptTemplate.id == template_id,
                SystemPromptTemplate.category == "doc_gen",
                SystemPromptTemplate.is_active == True,
            )
            .limit(1)
        )
        tmpl = session.execute(tmpl_stmt).first()
        if not tmpl:
            return None

        vars_stmt = (
            select(
                SystemPromptVariable.type,
                SystemPromptVariable.key,
                SystemPromptVariable.value,
                SystemPromptVariable.description,
            )
            .join(PromptMapping, PromptMapping.variable_id == SystemPromptVariable.id)
            .where(PromptMapping.template_id == template_id)
            .order_by(PromptMapping.id.asc())
        )
        var_rows = session.execute(vars_stmt).all()
        variables = [
            {
                "type": r.type,
                "key": r.key,
                "value": r.value,
                "description": r.description,
            }
            for r in var_rows
        ]

        return {
            "id": tmpl.id,
            "name": tmpl.name,
            "content": tmpl.content,
            "variables": variables,
        }