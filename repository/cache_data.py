from typing import Optional
from utils import logger
from utils.database import get_session
from sqlalchemy import select, desc
from storage.db_models import (
    CacheData
)

logger = logger(__name__)

def repo_set_cache(name: str, data: str, belongs_to: str = "global", by_id: Optional[int] = None) -> None:
    """
    cache_data 테이블에 캐시 데이터를 저장합니다.
    """
    with get_session() as session:
        try:
            cache_data = CacheData(
                name=name,
                data=data,
                belongs_to=belongs_to,
                by_id=by_id
            )
            session.add(cache_data)
            session.commit()
            logger.debug(f"Cache data saved: name={name}, belongs_to={belongs_to}")
        except Exception:
            session.rollback()
            logger.exception(f"Failed to save cache data: name={name}")
            raise


def repo_get_cache(name: str) -> Optional[str]:
    """
    cache_data 테이블에서 name으로 최신 캐시 데이터를 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(CacheData.data)
                .where(CacheData.name == name)
                .order_by(desc(CacheData.id))
                .limit(1)
            )
            result = session.execute(stmt).scalar()
            return result
        except Exception:
            logger.exception(f"Failed to get cache data: name={name}")
            return None
