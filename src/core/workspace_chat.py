from src.db.database import get_db
import json, re, time
from typing import Optional, List, Dict, Any
from src.utils import get_now_str


def get_chat_history(workspace_slug: str, session_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT prompt, response FROM workspace_chats
        WHERE api_session_id=? AND workspaceId=(
            SELECT id FROM workspaces WHERE slug=?
        )
        ORDER BY createdAt ASC
    """, (session_id, workspace_slug))
    rows = cursor.fetchall()
    messages = []
    for row in rows:
        messages.append({"role": "user", "content": row["prompt"]})
        if row["response"]:
            messages.append({"role": "assistant", "content": row["response"]})
    return messages


def save_chat(workspace_id: int, prompt: str, response: str, user_id: int, session_id: str, 
            sources: list = [], attachments: list = [], metrics: dict = {}):
    now = get_now_str()
    response_data = {
        "text":response,
        "source": sources,
        "type":"chat",
        "attatchments": attachments ,
        "metrics":metrics,
    }
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO workspace_chats (workspaceId, prompt, response, user_id, createdAt, lastUpdatedAt, api_session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (workspace_id, prompt, json.dumps(response_data, ensure_ascii=False), user_id, now, now, session_id))
    conn.commit()


def get_workspace_chat_history(slug: str, session_id: Optional[str] = None, 
                                limit: int = 20, order_by: str = "createdAt", 
                                order_direction: str = "ASC"):
    """워크스페이스 채팅 히스토리 조회 (API 응답용)"""
    conn = get_db()
    cursor = conn.cursor()
    
    # 기본 쿼리
    base_query = """
        SELECT prompt, response, createdAt FROM workspace_chats 
        WHERE workspaceId = (SELECT id FROM workspaces WHERE slug = ?)
    """
    params = [slug]
    
    # 세션 ID 필터링
    if session_id:
        base_query += " AND api_session_id = ?"
        params.append(session_id)
    
    # 정렬 및 제한
    base_query += f" ORDER BY {order_by} {order_direction} LIMIT {limit}"
    
    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    
    history = []
    for row in rows:
        # 메시지 시간을 Unix timestamp로 변환
        import time
        sent_at = int(time.mktime(time.strptime(row["createdAt"], "%Y-%m-%d %H:%M:%S")))
        
        # 사용자 메시지
        history.append({
            "role": "user",
            "content": row["prompt"],
            "sentAt": sent_at
        })
        
        # 어시스턴트 응답
        if row["response"]:
            try:
                response_data = json.loads(row["response"])
                response_text = response_data.get("text", row["response"])
            except (json.JSONDecodeError, TypeError):
                response_text = row["response"]
                
            history.append({
                "role": "assistant", 
                "content": response_text,
                "sentAt": sent_at + 1  # 응답은 1초 후로 설정
            })
    
    return history






    