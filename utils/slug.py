import re, uuid
from .database import get_db
from .logger import logger

logger = logger(__name__)

def re_slug(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = slug.strip('-')
    return slug

def generate_unique_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 slug 생성"""
    logger.info(f"Generating unique slug for: {name}")
    # 0) 한글/비ASCII 문자가 하나라도 있으면 UUID 강제
    if re.search(r"[^\x00-\x7F]", name):
        return str(uuid.uuid4())

    slug = re_slug(name)

	# 1) 정제 결과가 빈 값이거나 너무 짧으면 UUID
    if not slug or len(slug) < 2:
        return str(uuid.uuid4())

	# 2) 중복 체크 및 유니크하게 만들기
    conn = get_db()
    cursor = conn.cursor()
    try:
        original_slug = slug
        counter = 1
        while True:
            cursor.execute("SELECT id FROM workspaces WHERE slug=?", (slug,))
            if not cursor.fetchone():
                break
            slug = f"{original_slug}-{counter}"
            counter += 1
        return slug
    finally:
        conn.close()


def generate_thread_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 스레드 slug 생성"""
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
    conn = get_db()
    cursor = conn.cursor()
    original_slug = slug
    counter = 1
    while True:
        cursor.execute("SELECT id FROM workspace_threads WHERE slug=?", (slug,))
        if not cursor.fetchone():
            break
        slug = f"{original_slug}-{counter}"
        counter += 1
    conn.close()
    return slug