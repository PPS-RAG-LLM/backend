# /home/work/CoreIQ/yb/backend/repository/users/doc_gen_templates.py
from typing import List, Dict, Optional
from sqlalchemy import select, true
from sqlalchemy.exc import IntegrityError
from utils.database import get_session
from storage.db_models import SystemPromptTemplate, PromptMapping, SystemPromptVariable
from errors import DatabaseError
from utils import logger

logger = logger(__name__)
def repo_list_doc_gen_templates(default_only: bool) -> List[Dict[str, object]]:
    with get_session() as session:
        tmpl_stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
                SystemPromptTemplate.user_prompt,
            )
            .where(
                SystemPromptTemplate.category == "doc_gen",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        if default_only:
            tmpl_stmt = tmpl_stmt.where(SystemPromptTemplate.is_default == True)

        tmpl_rows = session.execute(tmpl_stmt).all()
        if not tmpl_rows:
            return []

        ids = [r.id for r in tmpl_rows]
        vars_stmt = (
            select(
                PromptMapping.template_id,
                SystemPromptVariable.id,
                SystemPromptVariable.type,
                SystemPromptVariable.key,
                SystemPromptVariable.value,
                SystemPromptVariable.description,
                SystemPromptVariable.required,
            )
            .join(SystemPromptVariable, SystemPromptVariable.id == PromptMapping.variable_id)
            .where(PromptMapping.template_id.in_(ids))
            .order_by(PromptMapping.id.asc())
        )
        vars_rows = session.execute(vars_stmt).all()
        by_tid: Dict[int, List[Dict[str, str]]] = {}
        for r in vars_rows:
            by_tid.setdefault(r.template_id, []).append(
                {"id": r.id, "type": r.type, "key": r.key, "value": r.value, "description": r.description, "required": r.required}
            )

        out: List[Dict[str, object]] = []
        for r in tmpl_rows:
            out.append({
                "id": r.id,
                "name": r.name,
                "system_prompt": r.system_prompt,
                "user_prompt": r.user_prompt,
                "variables": by_tid.get(r.id, []),
            })
        return out

def repo_get_doc_gen_template_by_id_with_vars(template_id: int) -> Optional[Dict[str, object]]:
    with get_session() as session:
        tmpl_stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
                SystemPromptTemplate.user_prompt,
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
                SystemPromptVariable.id,
                SystemPromptVariable.type,
                SystemPromptVariable.key,
                SystemPromptVariable.value,
                SystemPromptVariable.description,
                SystemPromptVariable.required,
            )
            .join(PromptMapping, PromptMapping.variable_id == SystemPromptVariable.id)
            .where(PromptMapping.template_id == template_id)
            .order_by(PromptMapping.id.asc())
        )
        var_rows = session.execute(vars_stmt).all()
        variables = [
            {"id": r.id, "type": r.type, "key": r.key, "value": r.value, "description": r.description, "required": r.required}
            for r in var_rows
        ]

        return {
            "id": tmpl.id,
            "name": tmpl.name,
            "system_prompt": tmpl.system_prompt,
            "user_prompt": tmpl.user_prompt,
            "variables": variables,
        }

def repo_create_doc_gen_template(
    name, system_prompt, user_prompt, variables
):
    with get_session() as session:
        try :
            template = SystemPromptTemplate(
                name= name,
                category = "doc_gen",
                system_prompt = system_prompt,
                user_prompt = user_prompt,
                is_default=False,
                is_active=True,
            )
            session.add(template)
            session.flush() # template.id 확보

            for var in variables or []:
                variable = SystemPromptVariable(
                    id = var["id"],
                    type = var["type"],  # null 오류
                    key = var["key"],
                    value = var.get("value"), # 유연
                    description=var["description"],
                    required=var.get("required", False),
                )
                # variable_id 세팅
                template.variable_mappings.append(PromptMapping(variable=variable))
            session.commit()
            session.refresh(template)

            return {
                "id": template.id,
                "name": template.name,
                "system_prompt": template.system_prompt,
                "user_prompt": template.user_prompt,
                "variables": variables or [],
            }
        except IntegrityError as exc:
                session.rollback()
                raise DatabaseError(f"QA Prompt template create failed: {exc}") from exc
    
def repo_update_doc_gen_template(
    template_id: int,
    name: str,
    system_prompt: str,
    user_prompt: Optional[str],
    variables: Optional[List[Dict[str, object]]],
) -> Optional[Dict[str, object]]:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "doc_gen":
            return None

        template.name = name
        template.system_prompt = system_prompt
        template.user_prompt = user_prompt

        existing_mappings = {mapping.variable_id: mapping for mapping in template.variable_mappings}
        seen_variable_ids: set[int] = set()

        for var in variables or []:
            var_id = var.get("id")
            if var_id is not None:
                seen_variable_ids.add(var_id)
                mapping = existing_mappings.pop(var_id, None)

                if not mapping:
                    variable = session.get(SystemPromptVariable, var_id)
                    if not variable:
                        raise DatabaseError(f"Variable not found: {var_id}")
                    mapping = PromptMapping(variable=variable)
                    template.variable_mappings.append(mapping)
                else:
                    variable = mapping.variable

                variable.type = var["type"]
                variable.key = var["key"]
                variable.value = var.get("value")
                variable.description = var["description"]
                variable.required = var.get("required", False)
            else:
                variable = SystemPromptVariable(
                    type=var["type"],
                    key=var["key"],
                    value=var.get("value"),
                    description=var["description"],
                    required=var.get("required", False),
                )
                template.variable_mappings.append(PromptMapping(variable=variable))

        for mapping in existing_mappings.values():
            variable = mapping.variable
            session.delete(mapping)

            if not variable:
                continue

            other_reference_exists = (
                session.execute(
                    select(PromptMapping.id)
                    .where(
                        PromptMapping.variable_id == variable.id,
                        PromptMapping.template_id != template.id,
                    )
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if not other_reference_exists:
                session.delete(variable)

        session.commit()
        session.refresh(template)

        return {
            "id": template.id,
            "name": template.name,
            "system_prompt": template.system_prompt,
            "user_prompt": template.user_prompt,
            "variables": variables or [],
        }

def repo_delete_doc_gen_template(template_id: int) -> bool:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "doc_gen":
            return False

        for mapping in list(template.variable_mappings):
            if mapping.variable:
                session.delete(mapping.variable)
            session.delete(mapping)

        session.delete(template)
        session.commit()
        return True

def repo_delete_doc_gen_prompt_variable(template_id: int, variable_id: int) -> bool:
    with get_session() as session:
        try:
            mapping = session.execute(
                    select(PromptMapping).where(
                        PromptMapping.template_id == template_id,
                        PromptMapping.variable_id == variable_id,
                    )
                ).scalars().first()
            if not mapping:
                return False
            other_refs = session.execute(
                    select(PromptMapping.id).where(
                        PromptMapping.variable_id == mapping.variable_id,
                        PromptMapping.id != mapping.id,
                    ).limit(1)
                ).scalars().first()

            if not other_refs and mapping.variable:
                session.delete(mapping.variable)

            session.delete(mapping)
            session.commit()
            return True
        except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"Doc gen prompt mapping delete failed: {exc}") from exc