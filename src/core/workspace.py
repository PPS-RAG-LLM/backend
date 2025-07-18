from src.db.database import get_db

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

def save_chat(workspace_id: int, prompt: str, response: str, user_id: int, session_id: str):
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO workspace_chats (workspaceId, prompt, response, user_id, createdAt, lastUpdatedAt, api_session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (workspace_id, prompt, response, user_id, now, now, session_id))
    conn.commit()