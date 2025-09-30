# /home/work/CoreIQ/yb/backend/repository/users/summary_templates.py
from typing import List, Dict, Optional, Any
from sqlalchemy import select
from utils.database import get_session
from storage.db_models import SystemPromptTemplate
from sqlalchemy.exc import IntegrityError
from errors import DatabaseError
from utils import logger

logger = logger(__name__)

def repo_list_summary_templates(default_only:bool) -> List[Dict[str, str]]:
    try:
        with get_session() as session:
            stmt = (
                select(
                    SystemPromptTemplate.id,
                    SystemPromptTemplate.name,
                    SystemPromptTemplate.system_prompt,
                    SystemPromptTemplate.user_prompt,

                )
                .where(
                    SystemPromptTemplate.category == "summary",
                    SystemPromptTemplate.is_active == True,
                )
                .order_by(SystemPromptTemplate.id.desc())
            )
            if default_only:
                stmt = stmt.where(SystemPromptTemplate.is_default == True)
            rows = session.execute(stmt).all()
            return [{"id": r.id, "name": r.name, "systemPrompt": r.system_prompt, "userPrompt": r.user_prompt} for r in rows]
    except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"Summary Prompt template create failed: {exc}") from exc
     

def repo_get_summary_template_by_id(template_id: int) -> Optional[Dict[str, str]]:
    try:
        with get_session() as session:
            stmt = (
                select(
                    SystemPromptTemplate.id,
                    SystemPromptTemplate.name,
                    SystemPromptTemplate.system_prompt,
                    SystemPromptTemplate.user_prompt,
                )
                .where(
                    SystemPromptTemplate.id == template_id,
                    SystemPromptTemplate.category == "summary",
                    SystemPromptTemplate.is_active == True,
                )
                .limit(1)
            )
            row = session.execute(stmt).first()
            if not row:
                return None
            logger.info("템플릿 찾음.")
            return {"id": row.id, "name": row.name, "systemPrompt": row.system_prompt, "userPrompt": row.user_prompt}
    except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"Summary Prompt template create failed: {exc}") from exc


def repo_create_summary_template(
    system_prompt: str, user_prompt: Optional[str]= ""
) -> Dict[str, Any]:
    """요약 시스템 프롬프트 생성"""
    with get_session() as session:
        try:
            template = SystemPromptTemplate(
                name="summary_prompt",
                category="summary",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                is_default=False,
                is_active=True,
            )
            session.add(template)
            session.commit()

            return {
                "id": int(template.id),
                "name": template.name,
                "systemPrompt": template.system_prompt,
                "isDefault": bool(template.is_default),
                "isActive": bool(template.is_active),
            }
        except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"Summary Prompt template create failed: {exc}") from exc

def repo_update_summary_template(
    template_id: int, system_prompt: str, user_prompt: Optional[str] = ""
    )-> Optional[Dict[str, str]]:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "summary":
            return None
        template.system_prompt = system_prompt
        template.user_prompt = user_prompt
        session.commit()
        session.refresh(template)
        return {
            "id": template.id,
            "name": template.name,
            "systemPrompt": template.system_prompt,
            "userPrompt": template.user_prompt 
        }

def repo_delete_summary_template(template_id: int) -> bool:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "summary":
            return False
        
        session.delete(template)
        session.commit()
        return True