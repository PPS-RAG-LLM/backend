import os, sqlite3, pathlib
from pathlib import Path
from config import config

# Resolve database paths from config with backward compatibility
_db_conf = config.get("database", {}) if isinstance(config, dict) else {}
_raw_db_path = _db_conf.get("db_path") or _db_conf.get("path") or "storage/pps_rag.db"
_raw_schema_path = _db_conf.get("schema_path") or _db_conf.get("schema") or "storage/schema.sql"

# Resolve to absolute paths relative to backend/ root
# NOTE: Use relative paths as-is so they resolve from the current working directory (backend/)
_DB_PATH = _raw_db_path
_SCHEMA_PATH = _raw_schema_path

# Ensure directory for DB exists
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

def _connect(db_path: str):
	conn = sqlite3.connect(db_path, check_same_thread=False)
	conn.row_factory = sqlite3.Row
	conn.execute("PRAGMA foreign_keys=ON;")
	conn.execute("PRAGMA journal_mode=WAL;")
	conn.execute("PRAGMA synchronous=NORMAL;")
	conn.execute("PRAGMA busy_timeout=5000;")
	return conn

def _init_db():
	# Apply schema if available
	con = _connect(_DB_PATH)
	try:
		try:
			sql_text = Path(_SCHEMA_PATH).read_text(encoding="utf-8")
			con.executescript(sql_text)
		except FileNotFoundError:
			# Schema file missing is tolerated; DB may already exist
			pass
		con.commit()
	finally:
		con.close()

# Initialize on import (idempotent)
_init_db()

def get_db():
	conn = _connect(_DB_PATH)
	return conn

