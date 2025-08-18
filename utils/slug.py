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
    slug = re_slug(name)
    
    # 한글이나 특수문자로 인해 빈 slug가 되는 경우 UUID 사용
    if not slug or len(slug) < 2:
        slug = str(uuid.uuid4())
        return slug  # UUID는 유니크하므로 중복 체크 불필요
    # 6. 중복 체크 및 유니크하게 만들기
    conn = get_db()
    cursor = conn.cursor()
    original_slug = slug
    counter = 1
    while True:
        cursor.execute("SELECT id FROM workspaces WHERE slug=?", (slug,))
        if not cursor.fetchone():
            break
        slug = f"{original_slug}-{counter}"
        counter += 1
    return slug

def generate_thread_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 스레드 slug 생성"""
    logger.info(f"Generating unique thread slug for: {name}")
    slug = re_slug(name)
    # 한글이나 특수문자로 인해 빈 slug가 되는 경우
    if not slug or len(slug) < 2:
        slug = f"thread-{str(uuid.uuid4())[:8]}"
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