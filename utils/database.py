import sqlite3, pathlib
from config import config

# SQLAlchemy 추가
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, declarative_base

db = config["database"]

# ORM 엔진/세션/베이스
DATABASE_URL = f"sqlite:///{db['path']}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()

_INITIALIZED = False

# SQLite 외래키 강제
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()


def init_db():
    """
    - ORM 모델(`storage/db_models.py`)만으로 테이블 생성(create_all)
    - `users`가 비어있으면 `storage/base.sql`을 실행해 기본 데이터 시드
    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    # 1) 모델 등록 및 테이블 생성
    import storage.db_models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # 2) 필요 시 기본 데이터 시드
    con = sqlite3.connect(db["path"])
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    # users 레코드 유무로 시드 필요 여부 판단(비어있으면 시드)
    try:
        cur.execute("SELECT COUNT(*) FROM users;")
        (user_count,) = cur.fetchone() or (0,)
    except sqlite3.OperationalError:
        # users 테이블이 없으면(이상 케이스), create_all 이후 다시 생성 시도
        con.close()
        Base.metadata.create_all(bind=engine)
        con = sqlite3.connect(db["path"])
        con.execute("PRAGMA foreign_keys=ON;")
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM users;")
        (user_count,) = cur.fetchone() or (0,)

    if user_count == 0:
        base_sql_path = pathlib.Path("storage/base.sql")
        if base_sql_path.exists():
            seed_sql = base_sql_path.read_text(encoding="utf-8")
            con.executescript(seed_sql)
            con.commit()

    con.close()

    _INITIALIZED = True


def get_session():
    """SQLAlchemy 세션을 반환(사용 후 close 필요). with 문 사용 권장."""
    return SessionLocal()


def get_db():
    """기존 sqlite3 연결 방식 - ORM 전환 중 호환성 유지용"""
    conn = sqlite3.connect(db["path"])
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn
