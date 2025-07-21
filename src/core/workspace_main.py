from src.db.database import get_db
import re, uuid
from typing import List, Dict, Any
from src.utils import get_now_str

def get_all_workspaces() -> List[Dict[str, Any]]:
    """모든 워크스페이스 목록 조회"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, slug, createdAt, lastUpdatedAt, 
               openAiPrompt, openAiHistory, openAiTemp,
               similarityThreshold, topN, chatMode, queryRefusalResponse
        FROM workspaces 
        ORDER BY createdAt DESC
    """)
    
    workspaces = []
    for row in cursor.fetchall():
        workspace = dict(row)
        workspace["threads"] = []  # 스레드 정보 (현재는 빈 배열)
        workspaces.append(workspace)
    
    return workspaces


def get_workspace_by_slug(workspace_slug: str):
    """slug로 워크스페이스 정보 조회"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, slug, createdAt, lastUpdatedAt, chatModel,
               openAiPrompt, similarityThreshold, topN, chatMode, 
               openAiTemp, openAiHistory, queryRefusalResponse
        FROM workspaces WHERE slug=?
    """, (workspace_slug,))
    
    result = cursor.fetchone()
    if result:
        return dict(result)
    return None


def get_workspace_id_from_slug(workspace_slug: str):
    conn=get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM workspaces WHERE slug=?", (workspace_slug,))
    result = cursor.fetchone()
    if result:
        return result["id"]
    raise ValueError(f"워크스페이스를 찾을 수 없습니다: {workspace_slug}")



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


def create_workspace(
    name: str, slug : str ="", created_by: int = 0, similarity_threshold: float = 0.5, open_ai_temp: float = 0.7,
    open_ai_history: int = 20, system_prompt: str = "" , query_refusal_response: str = "Custom refusal message",
    chat_mode: str = "chat", chat_model: str = "qwen-2.5-7b-instruct", top_n: int = 4
)-> Dict[str, Any]:
    """ 워크 스페이스 생성"""
    now = get_now_str()
    if not slug :
        slug = generate_unique_slug(name) # slug가 제공되지 않으면 이름에서 자동생성
    else:  # 제공된 slug가 이미 사용중인지 확인
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM workspaces WHERE slug=?", (slug,))
        if cursor.fetchone():
            raise ValueError(f"이미 사용중인 slug 입니다. : {slug}")
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO workspaces(
            name, slug, createdAt, lastUpdatedAt, chatModel, openAiPrompt, 
            similarityThreshold, topN, chatMode, openAiTemp, openAiHistory, queryRefusalResponse
        ) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,(
            name, slug, now, now, chat_model, system_prompt, 
            similarity_threshold, top_n, chat_mode, open_ai_temp, open_ai_history, query_refusal_response
        ))     

        workspace_id = cursor.lastrowid          # 생성된 워크스페이스 ID 반환
        conn.commit()

        # 생성된 워크스페이스 정보 반환
        cursor.execute("""          
           SELECT id, name, slug, createdAt, lastUpdatedAt, chatModel, openAiPrompt, 
                   similarityThreshold, topN, chatMode, openAiTemp, openAiHistory, queryRefusalResponse
            FROM workspaces
            WHERE id = ?
        """, (workspace_id,))
        result = cursor.fetchone()
        return dict(result)

    except Exception as e:
        conn.rollback()
        raise Exception(f"워크스페이스 생성 실패: {str(e)}")


def delete_workspace(workspace_slug: str):
    """slug로 워크스페이스 삭제"""
    conn = get_db()
    cursor = conn.cursor()
    try:
        # 1. 워크 스페이스가 존재하는지 확인
        cursor.execute("SELECT id FROM workspaces WHERE slug=?", (workspace_slug,))
        workspace = cursor.fetchone()
        if not workspace:
            raise ValueError(f"워크스페이스를 찾을 수 없습니다: {workspace_slug}")

        workspace_id = workspace["id"]

        # 2. 관련 데이터 삭제 (Foreign Key CASCADE로 자동 삭제되지만 명시적으로 처리)
        # workspace_chats, workspace_users, workspace_documents 등은 
        # schema.sql에서 CASCADE로 설정되어 있어서 자동 삭제됨
        
        # 3. 워크스페이스 삭제
        cursor.execute("DELETE FROM workspaces WHERE slug=?", (workspace_slug,))
        if cursor.rowcount == 0:
            raise ValueError(f"워크스페이스 삭제에 실패했습니다: {workspace_slug}")
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        raise Exception(f"워크스페이스 삭제 중 오류 발생: {str(e)}")

def update_workspace(slug: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """워크스페이스 정보 업데이트"""
    conn = get_db()
    cursor = conn.cursor()
    
    # 워크스페이스 존재 확인
    cursor.execute("SELECT id FROM workspaces WHERE slug=?", (slug,))
    workspace = cursor.fetchone()
    if not workspace:
        raise ValueError(f"워크스페이스를 찾을 수 없습니다: {slug}")
    
    # 업데이트할 필드와 값 준비
    set_clauses = []
    params = []
    
    for key, value in update_data.items():
        if key == "name":
            set_clauses.append("name = ?")
            params.append(value)
        elif key == "openAiTemp":
            set_clauses.append("openAiTemp = ?")
            params.append(value)
        elif key == "openAiHistory":
            set_clauses.append("openAiHistory = ?")
            params.append(value)
        elif key == "openAiPrompt":
            set_clauses.append("openAiPrompt = ?")
            params.append(value)
    
    if not set_clauses:
        raise ValueError("업데이트할 필드가 없습니다")
    
    # lastUpdatedAt 추가
    set_clauses.append("lastUpdatedAt = ?")
    params.append(get_now_str())
    params.append(slug)
    
    try:
        # 업데이트 실행
        query = f"UPDATE workspaces SET {', '.join(set_clauses)} WHERE slug = ?"
        cursor.execute(query, params)
        conn.commit()
        
        # 업데이트된 워크스페이스 정보 반환
        updated_workspace = get_workspace_by_slug(slug)
        if updated_workspace:
            updated_workspace["documents"] = []  # 문서 정보 추가
            return updated_workspace
        else:
            raise Exception("업데이트된 워크스페이스를 찾을 수 없습니다")
            
    except Exception as e:
        conn.rollback()
        raise Exception(f"워크스페이스 업데이트 실패: {str(e)}")

