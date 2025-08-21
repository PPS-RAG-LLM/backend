from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))

def now_kst_string() -> str:
    """현재 한국시간을 SQLite용 문자열로 반환"""
    return now_kst().strftime("%Y-%m-%d %H:%M:%S")

def expires_at_kst() -> str:
    return (now_kst() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def to_kst(value):
    """입력된 datetime 또는 문자열을 KST(Asia/Seoul)로 변환해 문자열로 반환한다.

    허용되는 입력 형식:
    1) `datetime` 객체 (naive 또는 timezone-aware)
    2) 문자열 "YYYY-MM-DD HH:MM:SS" (SQLite 기본 형식, naive/UTC)
    3) ISO 8601 문자열 (Python `datetime.fromisoformat` 지원 형식)
    """
    if value is None:
        return None

    if isinstance(value, str):
        try:
            dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # fallback to ISO 8601 parsing
            dt = datetime.fromisoformat(value)
    elif isinstance(value, datetime):
        dt = value
    else:
        raise TypeError("to_kst: value must be datetime or str")

    # If naive, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))

    return dt.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")