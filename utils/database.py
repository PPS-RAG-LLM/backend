import sqlite3, pathlib
from config import config

db = config["database"]

_INITIALIZED = False

def init_db():
	global _INITIALIZED
	if _INITIALIZED:
		return
	con = sqlite3.connect(db["path"])
	con.execute("PRAGMA foregin_keys=ON;")
	cur = con.cursor()

	# 대표 테이블 존재 여부 확인 (users 또는 workspaces 등)
	cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
	exists= cur.fetchone() is not None
	if not exists:
		sql = pathlib.Path(db["schema_path"]).read_text(encoding="utf-8")
		con.executescript(sql)
		con.commit()
	con.close()
	_INITIALIZED = True
	

def get_db():
	conn = sqlite3.connect(db["path"])
	conn.row_factory = sqlite3.Row
	conn.execute("PRAGMA foreign_keys=ON;")
	return conn

