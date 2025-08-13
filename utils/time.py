from datetime import datetime
from zoneinfo import ZoneInfo

def to_kst(dt_str: str) -> str:
    # SQLite 기본 DATETIME("YYYY-MM-DD HH:MM:SS")를 UTC로 간주
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return dt.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
