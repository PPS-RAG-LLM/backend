# /home/work/CoreIQ/yb/backend/repository/users/summary_templates.py
from typing import List, Dict, Optional
from sqlalchemy import select
from utils.database import get_session
from storage.db_models import SystemPromptTemplate

def repo_list_summary_templates() -> List[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
            )
            .where(
                SystemPromptTemplate.category == "summary",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        rows = session.execute(stmt).all()
        return [{"id": r.id, "name": r.name, "system_prompt": r.system_prompt} for r in rows]

def repo_get_summary_template_by_id(template_id: int) -> Optional[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
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
        return {"id": row.id, "name": row.name, "system_prompt": row.system_prompt}