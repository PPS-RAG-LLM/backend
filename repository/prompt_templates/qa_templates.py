# /home/work/CoreIQ/yb/backend/repository/users/summary_templates.py
from typing import List, Dict, Any, Optional
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from utils.database import get_session
from storage.db_models import SystemPromptTemplate
from errors import DatabaseError

def repo_list_qna_templates() -> List[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
                SystemPromptTemplate.user_prompt,
            )
            .where(
                SystemPromptTemplate.category == "qna",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        rows = session.execute(stmt).all()
        return [{"id": r.id, "name": r.name, "systemPrompt": r.system_prompt, "userPrompt":r.user_prompt} for r in rows]
        

def repo_create_qna_template(
    system_prompt: str, user_prompt: Optional[str]=""
) -> Dict[str, Any]:
    with get_session() as session:
        """QnA 시스템프롬프트 생성"""
        try:
            template = SystemPromptTemplate(
                name="qna_prompt",
                category="qna",
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
                "system_prompt": template.system_prompt,
                "is_default": bool(template.is_default),
                "is_active": bool(template.is_active),
            }
        except IntegrityError as exc:
            session.rollback()
            raise DatabaseError(f"QA Prompt template create failed: {exc}") from exc


def repo_update_qna_template(
    template_id: int, system_prompt: str, user_prompt: Optional[str] = ""
    )-> Optional[Dict[str, str]]:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "qna":
            return None
        template.system_prompt = system_prompt
        template.user_prompt = user_prompt
        session.commit()
        session.refresh(template)
        return {
            "id": template.id,
            "name": template.name,
            "system_prompt": template.system_prompt,
            "user_prompt": template.user_prompt 
        }

def repo_delete_qna_template(template_id: int) -> bool:
    with get_session() as session:
        template = session.get(SystemPromptTemplate, template_id)
        if not template or template.category != "qna":
            return False
        
        session.delete(template)
        session.commit()
        return True