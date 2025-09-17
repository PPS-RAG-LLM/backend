from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def now_kst() -> datetime:
    """데이터베이스 저장시 사용하는 현재 한국시간"""
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)


def now_kst_string() -> str:
    """현재 한국시간을 SQLite용 문자열로 반환"""
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def to_kst_string(dt: datetime) -> str:
    """데이터베이스에서 조회한한 datetime을 KST 기준의 'YYYY-MM-DD HH:MM:SS' 문자열로 변환"""
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
