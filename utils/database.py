import pathlib
from config import config

# SQLAlchemy 추가
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. 설정 가져오기 (config.yaml에 url이 있다고 가정)
# 만약 config 구조가 다르다면 적절히 수정하세요.
# 예: DATABASE_URL = f"postgresql://{user}:{pw}@{host}:{port}/{db_name}"
db_conf = config.get("database", {})
DATABASE_URL = db_conf.get("url")

# 2. 엔진 생성 (SQLite 전용 옵션 제거)
engine = create_engine(
    DATABASE_URL,
    pool_size=20,        # 커넥션 풀 크기 설정
    max_overflow=0,
    pool_pre_ping=True,  # 연결 끊김 방지
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)
Base = declarative_base()



def init_db():
    """
    테이블을 생성하고, 데이터가 비어있으면 초기 데이터를 적재합니다.
    """
    # 1) 테이블 생성
    import storage.db_models  # 모델 등록
    Base.metadata.create_all(bind=engine)

    # 2) 데이터 시딩 (SQLAlchemy 엔진 사용)
    with engine.connect() as conn:
        # users 테이블 확인
        # Postgres에서는 테이블명을 따옴표로 감쌀 때 대소문자 주의 ("users")
        result = conn.execute(text('SELECT COUNT(*) FROM "users"'))
        user_count = result.scalar()

        if user_count == 0:
            base_sql_path = pathlib.Path("storage/base.sql")
            if base_sql_path.exists():
                print("초기 데이터를 적재합니다...")
                seed_sql = base_sql_path.read_text(encoding="utf-8")
                
                # 주의: base.sql 파일 내의 문법이 Postgres와 호환되어야 함
                # [수정] SQLAlchemy의 text()는 콜론(:)을 변수로 인식하므로,
                # raw_connection을 사용하여 직접 실행합니다.
                raw_conn = engine.raw_connection()
                try:
                    cursor = raw_conn.cursor()
                    cursor.execute(seed_sql)
                    raw_conn.commit()
                    print("✅ 초기 데이터 적재 완료")
                finally:
                    raw_conn.close()

def get_session():
    """새로운 SQLAlchemy 세션 반환"""
    return SessionLocal()
