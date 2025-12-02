from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy.orm import Session

from storage.db_models import SecurityLevelConfigTask, SecurityLevelKeywordsTask
from utils.database import get_session
from utils.time import now_kst

def get_security_config_by_task_type(task_type: str) -> Optional[SecurityLevelConfigTask]:
    """
    특정 작업 유형(task_type)에 대한 보안 레벨 설정을 조회합니다.
    """
    with get_session() as session:
        return (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )

def get_security_keywords_by_task_type(task_type: str) -> List[Tuple[int, str]]:
    """
    특정 작업 유형(task_type)에 대한 보안 레벨별 키워드 목록을 조회합니다.
    Returns: List of (level, keyword) tuples
    """
    with get_session() as session:
        rows = (
            session.query(
                SecurityLevelKeywordsTask.level, SecurityLevelKeywordsTask.keyword
            )
            .filter(SecurityLevelKeywordsTask.task_type == task_type)
            .order_by(
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        return [(int(r.level), str(r.keyword)) for r in rows]

def upsert_security_config_and_keywords(
    task_type: str, max_level: int, levels_map: Dict[int, List[str]]
) -> None:
    """
    특정 작업 유형의 보안 레벨 설정(max_level)을 업데이트하고,
    해당 작업 유형의 모든 키워드를 삭제한 후 새로 등록합니다(Transaction).
    """
    with get_session() as session:
        # 1. Config Upsert
        cfg = (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )
        if not cfg:
            cfg = SecurityLevelConfigTask(task_type=task_type, max_level=int(max_level))
            session.add(cfg)
        else:
            cfg.max_level = int(max_level)
            cfg.updated_at = now_kst()
        
        # 2. Keywords Replace (Delete All -> Insert New)
        session.query(SecurityLevelKeywordsTask).filter(
            SecurityLevelKeywordsTask.task_type == task_type
        ).delete()
        
        for lv, kws in levels_map.items():
            for kw in kws:
                session.add(
                    SecurityLevelKeywordsTask(
                        task_type=task_type, level=int(lv), keyword=str(kw)
                    )
                )
        session.commit()

def get_all_security_configs_and_keywords() -> Tuple[List[Tuple[str, int]], List[Tuple[str, int, str]]]:
    """
    모든 작업 유형의 설정과 키워드를 한 번에 조회합니다.
    Returns:
        (configs, keywords)
        configs: List of (task_type, max_level)
        keywords: List of (task_type, level, keyword)
    """
    with get_session() as session:
        configs = session.query(
            SecurityLevelConfigTask.task_type, SecurityLevelConfigTask.max_level
        ).all()
        
        keywords = (
            session.query(
                SecurityLevelKeywordsTask.task_type,
                SecurityLevelKeywordsTask.level,
                SecurityLevelKeywordsTask.keyword,
            )
            .order_by(
                SecurityLevelKeywordsTask.task_type.asc(),
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        
        # Convert to simple types to detach from session
        config_list = [(str(c.task_type), int(c.max_level)) for c in configs]
        keyword_list = [(str(k.task_type), int(k.level), str(k.keyword)) for k in keywords]
        
        return config_list, keyword_list

