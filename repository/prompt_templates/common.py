from typing import Optional, Tuple
from sqlalchemy import select, func
from utils.database import get_session
from storage.db_models import SystemPromptTemplate

def repo_find_default_template(category: str, subcategory: Optional[str] = None) -> Optional[Tuple[int, str]]:
    """
    주어진 카테고리(및 서브카테고리)에 해당하는 기본(Active & Default) 템플릿을 찾습니다.
    """
    with get_session() as session:
        stmt = select(SystemPromptTemplate.id, SystemPromptTemplate.name).where(
            SystemPromptTemplate.category == category,
            func.coalesce(SystemPromptTemplate.is_default, False) == True,
            func.coalesce(SystemPromptTemplate.is_active, True) == True
        )
        
        if subcategory:
             # lower(name) == subcategory comparison
             stmt = stmt.where(func.lower(SystemPromptTemplate.name) == subcategory.lower())
             
        stmt = stmt.order_by(SystemPromptTemplate.id.desc()).limit(1)
        row = session.execute(stmt).first()
        if row:
            return row.id, row.name
        return None