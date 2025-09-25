# /home/work/CoreIQ/yb/backend/repository/users/summary_templates.py
from typing import List, Dict
from sqlalchemy import select
from utils.database import get_session
from storage.db_models import SystemPromptTemplate

def repo_list_qa_templates() -> List[Dict[str, str]]:
    with get_session() as session:
        stmt = (
            select(
                SystemPromptTemplate.id,
                SystemPromptTemplate.name,
                SystemPromptTemplate.system_prompt,
            )
            .where(
                SystemPromptTemplate.category == "qa",
                SystemPromptTemplate.is_active == True,
            )
            .order_by(SystemPromptTemplate.id.desc())
        )
        rows = session.execute(stmt).all()
        return [{"id": r.id, "name": r.name, "system_prompt": r.system_prompt} for r in rows]
        
