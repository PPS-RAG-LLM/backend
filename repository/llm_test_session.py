

from typing import Optional
from storage.db_models import LlmTestSession
from utils.database import get_session


def insert_test_session(sid: str, sess_dir: str, collection: str):
    with get_session() as session:
        obj = LlmTestSession(
            sid=sid,
            directory=str(sess_dir),
            collection=collection,
        )
        session.add(obj)
        session.commit()
        session.refresh(obj)
    return obj


def get_test_session_by_sid(sid: str) -> Optional[LlmTestSession]:
    with get_session() as session:
        row = session.query(LlmTestSession).filter(LlmTestSession.sid == sid).first()
        if not row:
            return None
        return row