import re, uuid, unicodedata
from pathlib import Path
from sqlalchemy import select
from .database import get_session
from .logger import logger

# 지연 import를 위해 타입 힌트용으로만 쓰거나 함수 내 import 사용 권장
# from storage.db_models import Workspace, WorkspaceThread

logger = logger(__name__)

def re_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = slug.strip('-')
    return slug

def generate_unique_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 slug 생성"""
    from storage.db_models import Workspace  # 순환 참조 방지

    logger.info(f"Generating unique slug for: {name}")
    # 0) 한글/비ASCII 문자가 하나라도 있으면 UUID 강제
    if re.search(r"[^\x00-\x7F]", name):
        return str(uuid.uuid4())

    slug = re_slug(name)

    # 1) 정제 결과가 빈 값이거나 너무 짧으면 UUID
    if not slug or len(slug) < 2:
        return str(uuid.uuid4())

    # 2) 중복 체크 및 유니크하게 만들기
    try:
        with get_session() as session:
            original_slug = slug
            counter = 1
            while True:
                stmt = select(Workspace.id).where(Workspace.slug == slug).limit(1)
                result = session.execute(stmt).first()
                if not result:
                    break
                slug = f"{original_slug}-{counter}"
                counter += 1
            return slug
    except Exception as e:
        logger.error(f"Error generating slug: {e}")
        # DB 에러 시 안전하게 UUID 반환
        return str(uuid.uuid4())


def generate_thread_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 스레드 slug 생성"""
    from storage.db_models import WorkspaceThread  # 순환 참조 방지

    logger.info(f"Generating unique thread slug for: {name}")
    # 0) 한글/비ASCII 문자가 하나라도 있으면 UUID 강제
    if re.search(r"[^\x00-\x7F]", name):
        return str(uuid.uuid4())
    slug = re_slug(name)
    logger.info(f"slug: {slug}")
    # 한글이나 특수문자로 인해 빈 slug가 되는 경우
    if not slug or len(slug) < 2:
        slug = str(uuid.uuid4())
        return slug
    
    # workspace_threads 테이블에서 중복 체크
    try:
        with get_session() as session:
            original_slug = slug
            counter = 1
            while True:
                stmt = select(WorkspaceThread.id).where(WorkspaceThread.slug == slug).limit(1)
                result = session.execute(stmt).first()
                if not result:
                    break
                slug = f"{original_slug}-{counter}"
                counter += 1
            return slug
    except Exception as e:
        logger.error(f"Error generating thread slug: {e}")
        return str(uuid.uuid4())


### 문서 파일명 안전하게 줄이기 ###


MAX_SAFE_STEM = 80

def make_safe_filename(original: str, doc_id: str, suffix: str) -> str:
    """긴/특수문자 파일명을 안전하게 줄여주는 헬퍼"""
    stem = Path(original).stem
    stem = unicodedata.normalize("NFKD", stem)
    stem = re.sub(r"\s+", "_", stem).strip("_")
    stem = re.sub(r"[^\w\-가-힣]", "", stem)
    if not stem:
        stem = "document"
    if len(stem) > MAX_SAFE_STEM:
        stem = stem[:MAX_SAFE_STEM]
    return f"{stem}-{doc_id}.{suffix}"
