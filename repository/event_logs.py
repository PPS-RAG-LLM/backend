from typing import Optional
from utils import logger
from utils.database import get_session
from storage.db_models import EventLog

logger = logger(__name__)

def repo_add_event_log(event: str, metadata: str, user_id: Optional[int] = None) -> bool:
    """
    event_logs 테이블에 새로운 로그를 추가합니다.
    """
    with get_session() as session:
        try:
            log_entry = EventLog(
                event=event,
                metadata_json=metadata,
                user_id=user_id
            )
            session.add(log_entry)
            session.commit()
            return True
        except Exception:
            session.rollback()
            logger.exception("Failed to add event log")
            return False

