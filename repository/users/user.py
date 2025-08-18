from utils import get_db

def get_user_by_username(username: str) -> dict:
    try:
        db = get_db()
        user = db.execute(
            "SELECT id, username, name, department, position, security_level, password FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        return user

    finally:
        db.close()

