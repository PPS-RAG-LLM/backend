import sqlite3, pathlib
from config import config

db = config["database"]
sql = pathlib.Path(db["schema_path"]).read_text(encoding="utf-8")
con = sqlite3.connect(db["db_path"])
con.execute("PRAGMA foreign_keys=ON;")
con.executescript(sql)
con.commit()
con.close()

def get_db():
	conn = sqlite3.connect(db["db_path"])
	conn.row_factory = sqlite3.Row
	conn.execute("PRAGMA foreign_keys=ON;")
	return conn

