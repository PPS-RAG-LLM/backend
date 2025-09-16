from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def now_kst_string() -> str:
    """현재 한국시간을 SQLite용 문자열로 반환"""
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")


def to_kst_string(dt: datetime) -> str:
    """주어진 datetime을 KST 기준의 'YYYY-MM-DD HH:MM:SS' 문자열로 변환"""
    if not isinstance(dt, datetime):
        return str(dt)
    # tz 정보가 없으면 UTC로 간주 후 KST로 변환
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")