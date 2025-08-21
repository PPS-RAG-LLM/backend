from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def now_kst_string() -> str:
    """현재 한국시간을 SQLite용 문자열로 반환"""
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

def expires_at_kst() -> str:
    return (now_kst() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")