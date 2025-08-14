import re, uuid
from utils.database import get_db

def generate_unique_slug(name: str) -> str:
    """워크스페이스 이름에서 고유한 slug 생성"""
    slug = name.lower().strip()                # 1. 소문자 변환 및 공백을 하이픈으로 변경
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)     # 2. 특수문자 제거 (영문, 숫자, 하이픈만 허용)
    slug = re.sub(r'-+', '-', slug)            # 3. 연속된 하이픈 제거
    slug = slug.strip('-')                     # 4. 앞뒤 하이픈 제거
    
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
