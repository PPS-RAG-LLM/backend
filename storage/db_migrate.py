#!/usr/bin/env python3
import argparse
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]  # .../backend
sys.path.insert(0, str(ROOT))
from config import config

BACKUP_DIR = ROOT / "storage" / ".backup"


def ensure_backup_dir():
	BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def timestamp() -> str:
	return datetime.now().strftime("%Y-%m-%d_%H%M%S")

def get_db_paths() -> tuple[Path, Path]:
	db_path = (ROOT / config["database"]["db_path"]).resolve()
	schema_path = (ROOT / config["database"]["schema_path"]).resolve()
	return db_path, schema_path

def backup_data_only_dump(db_path: Path) -> Path:
	ensure_backup_dir()
	out_file = BACKUP_DIR / f"pps_rag_data_{timestamp()}.sql"
	with sqlite3.connect(db_path) as con, open(out_file, "w", encoding="utf-8") as f:
		for line in con.iterdump():
			if line.startswith("INSERT INTO"):
				f.write(line + "\n")
	print(f"[+] 데이터 전용 덤프 생성: {out_file}")
	return out_file

def file_backup(db_path: Path) -> Path:
	ensure_backup_dir()
	dst = BACKUP_DIR / f"pps_rag_{timestamp()}.db"
	with sqlite3.connect(db_path) as src, sqlite3.connect(dst) as dest:
		src.backup(dest)
	print(f"[+] 파일 백업 생성: {dst}")
	return dst

def recreate_db(schema_path: Path, db_path: Path):
	if db_path.exists():
		db_path.unlink()
	with sqlite3.connect(db_path) as con:
		sql = schema_path.read_text(encoding="utf-8")
		con.execute("PRAGMA foreign_keys=ON;")
		con.executescript(sql)
		con.commit()
	print(f"[+] 스키마로 신규 DB 생성 완료: {db_path}")

def restore_data_from_dump(db_path: Path, dump_sql: Path):
	with sqlite3.connect(db_path) as con:
		con.execute("PRAGMA foreign_keys=OFF;")
		sql = dump_sql.read_text(encoding="utf-8")
		con.executescript(sql)
		con.execute("PRAGMA foreign_keys=ON;")
		violations = list(con.execute("PRAGMA foreign_key_check;"))
		if violations:
			print(f"[!] 외래키 위반 {len(violations)}건 감지:", file=sys.stderr)
			for row in violations[:10]:
				print(f"    {row}", file=sys.stderr)
			if len(violations) > 10:
				print("    ...", file=sys.stderr)
			sys.exit(1)
	print(f"[+] 데이터 복원 완료: {dump_sql}")

def latest_dump_file() -> Path | None:
	if not BACKUP_DIR.exists():
		return None
	candidates = sorted(BACKUP_DIR.glob("pps_rag_data_*.sql"), key=lambda p: p.stat().st_mtime, reverse=True)
	return candidates[0] if candidates else None

def do_backup():
	db_path, _ = get_db_paths()
	backup_data_only_dump(db_path)
	file_backup(db_path)

def do_recreate():
	db_path, schema_path = get_db_paths()
	recreate_db(schema_path, db_path)

def do_restore(dump: str | None):
	db_path, _ = get_db_paths()
	if dump:
		dump_path = Path(dump).resolve()
	else:
		dump_path = latest_dump_file()
		if not dump_path:
			print("[!] 사용할 덤프 파일을 찾을 수 없습니다. --dump 로 지정하세요.", file=sys.stderr)
			sys.exit(2)
	restore_data_from_dump(db_path, dump_path)

def do_full(dump: str | None):
	db_path, schema_path = get_db_paths()
	# 1) 백업
	data_dump = backup_data_only_dump(db_path)
	file_backup(db_path)
	# 2) 재생성
	recreate_db(schema_path, db_path)
	# 3) 복원 (우선순위: 사용자가 지정한 dump -> 방금 생성한 dump)
	restore_data_from_dump(db_path, Path(dump).resolve() if dump else data_dump)

def main():
	parser = argparse.ArgumentParser(description="SQLite 백업/재생성/복원 자동화")
	sub = parser.add_subparsers(dest="cmd", required=True)

	sub.add_parser("backup", help="데이터 전용 SQL + 파일 백업(.db) 생성")
	sub.add_parser("recreate", help="스키마로 새 DB 생성(기존 파일 삭제)")
	restore_p = sub.add_parser("restore", help="덤프 SQL로 데이터 복원")
	restore_p.add_argument("--dump", type=str, help="복원에 사용할 .sql 경로(미지정 시 최신 덤프)")
	full_p = sub.add_parser("full", help="백업 -> 재생성 -> 복원 전체 수행")
	full_p.add_argument("--dump", type=str, help="복원에 사용할 .sql 경로(미지정 시 방금 만든 덤프 사용)")

	args = parser.parse_args()

	match args.cmd:
		case "backup":
			do_backup()
		case "recreate":
			do_recreate()
		case "restore":
			do_restore(args.dump)
		case "full":
			do_full(args.dump)
		case _:
			parser.print_help()
			sys.exit(2)

if __name__ == "__main__":
	main()
# 
"""
# 전체 수행(백업 → 재생성 → 복원):
>>> python storage/db_migrate.py full

# 백업만 수행:
>>> python storage/db_migrate.py backup

# 스키마로 새 DB만 재생성
>>> python storage/db_migrate.py recreate

# 특정 dump 파일로 복원
>>> python storage/db_migrate.py restore --dump /home/work/CoreIQ/Ruah/backend/storage/.backup/pps_rag_data_YYYY-MM-DD_HHMMSS.sql
"""