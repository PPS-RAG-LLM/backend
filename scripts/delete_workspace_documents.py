# scripts/delete_workspace_documents.py
import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 추가 (backend/에서 실행하지 않아도 import 가능)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

# 파일 저장 위치(존재하지 않을 수도 있어 기본값 준비)
SAVE_DOC_DIR = config.get("user_documents", {})
DOC_INFO_DIR = Path(SAVE_DOC_DIR.get("doc_info_dir", "storage/documents/documents-info"))
VEC_CACHE_DIR = Path(SAVE_DOC_DIR.get("vector_cache_dir", "storage/documents/vector-cache"))

# 선택: --db-path로 오버라이드 가능
DB_PATH_OVERRIDE: Path | None = None


def _resolve_db_path() -> Path:
    db_cfg = Path(config["database"]["path"])
    if db_cfg.is_absolute():
        return db_cfg
    return (ROOT / db_cfg).resolve()


def _connect_db() -> sqlite3.Connection:
    db_path = DB_PATH_OVERRIDE or _resolve_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_docs_by_ids(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, sqlite3.Row]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(
        f"""
        SELECT doc_id, filename, docpath
        FROM workspace_documents
        WHERE doc_id IN ({qmarks})
        """,
        doc_ids,
    )
    return {r["doc_id"]: r for r in cur.fetchall()}


def _fetch_all_docs(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.execute(
        """
        SELECT doc_id, filename, docpath
        FROM workspace_documents
        ORDER BY id DESC
        """
    )
    return list(cur.fetchall())


def _count_vectors(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) FROM document_vectors")
    return int(cur.fetchone()[0])


def _count_vectors_for(conn: sqlite3.Connection, doc_ids: List[str]) -> int:
    if not doc_ids:
        return 0
    qmarks = ",".join(["?"] * len(doc_ids))
    cur = conn.execute(f"SELECT COUNT(*) FROM document_vectors WHERE doc_id IN ({qmarks})", doc_ids)
    return int(cur.fetchone()[0])


def _delete_vectors_by_doc_ids(conn: sqlite3.Connection, doc_ids: List[str]) -> int:
    if not doc_ids:
        return 0
    before = _count_vectors_for(conn, doc_ids)
    qmarks = ",".join(["?"] * len(doc_ids))
    deleted = conn.execute(
        f"DELETE FROM document_vectors WHERE doc_id IN ({qmarks})",
        doc_ids,
    ).rowcount
    conn.commit()
    after = _count_vectors_for(conn, doc_ids)
    print(f"[vectors] before={before}, deleted={deleted}, after={after}")
    return deleted


def _delete_all_vectors(conn: sqlite3.Connection) -> int:
    before = _count_vectors(conn)
    deleted = conn.execute("DELETE FROM document_vectors").rowcount
    conn.commit()
    after = _count_vectors(conn)
    print(f"[vectors] before={before}, deleted={deleted}, after={after}")
    return deleted


def _delete_db_records(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, int]:
    if not doc_ids:
        return {"workspace_documents": 0, "document_vectors": 0}
    qmarks = ",".join(["?"] * len(doc_ids))
    # document_vectors(1:N) 먼저 삭제
    dv = conn.execute(
        f"DELETE FROM document_vectors WHERE doc_id IN ({qmarks})",
        doc_ids,
    ).rowcount
    # workspace_documents 삭제
    wd = conn.execute(
        f"DELETE FROM workspace_documents WHERE doc_id IN ({qmarks})",
        doc_ids,
    ).rowcount
    conn.commit()
    return {"workspace_documents": wd, "document_vectors": dv}


def _remove_files(filename: str, doc_id: str, doc_rel_path: str, dry_run: bool = True) -> Dict[str, bool]:
    # documents-info/<filename>-<doc_id>.json
    info_path = DOC_INFO_DIR / f"{filename}-{doc_id}.json"
    # vector-cache/<doc_id>.json
    vec_path = VEC_CACHE_DIR / f"{doc_id}.json"

    removed = {"doc_info": False, "vector_cache": False}

    for p, key in ((info_path, "doc_info"), (vec_path, "vector_cache")):
        if p.exists():
            if dry_run:
                print(f"[dry-run] would remove: {p}")
            else:
                try:
                    p.unlink()
                    removed[key] = True
                    print(f"[removed] {p}")
                except Exception as e:
                    print(f"[warn] failed to remove {p}: {e}")

    return removed


def delete_by_doc_ids(doc_ids: List[str], dry_run: bool = True, vectors_only: bool = False) -> None:
    conn = _connect_db()
    try:
        if vectors_only:
            if dry_run:
                print(f"[dry-run] would delete document_vectors for doc_ids: {doc_ids}")
            else:
                _delete_vectors_by_doc_ids(conn, doc_ids)
            return

        rows = _fetch_docs_by_ids(conn, doc_ids)

        # 파일 삭제(문서 메타가 있는 것만)
        if rows:
            for doc_id in doc_ids:
                row = rows.get(doc_id)
                if not row:
                    print(f"[skip] doc_id not found in workspace_documents: {doc_id}")
                    continue
                _remove_files(
                    filename=row["filename"],
                    doc_id=row["doc_id"],
                    doc_rel_path=row["docpath"] or "",
                    dry_run=dry_run,
                )
        else:
            print("[info] no matching workspace_documents; will still delete document_vectors for provided doc_ids.")

        # DB 삭제
        if dry_run:
            print(f"[dry-run] would delete DB rows(workspace_documents + document_vectors) for: {doc_ids}")
        else:
            counts = _delete_db_records(conn, doc_ids)
            print(f"[db-removed] {counts}")
    finally:
        conn.close()


def delete_all(dry_run: bool = True, vectors_only: bool = False) -> None:
    conn = _connect_db()
    try:
        if vectors_only:
            if dry_run:
                total = _count_vectors(conn)
                print(f"[dry-run] would delete ALL rows from document_vectors (current={total})")
            else:
                _delete_all_vectors(conn)
            return

        rows = _fetch_all_docs(conn)
        if not rows:
            print("[info] workspace_documents is empty.")
            return

        doc_ids = [r["doc_id"] for r in rows]

        # 파일 삭제
        for r in rows:
            _remove_files(
                filename=r["filename"],
                doc_id=r["doc_id"],
                doc_rel_path=r["docpath"] or "",
                dry_run=dry_run,
            )

        # DB 삭제
        if dry_run:
            print(f"[dry-run] would delete ALL DB rows(workspace_documents + document_vectors): total={len(doc_ids)}")
        else:
            counts = _delete_db_records(conn, doc_ids)
            print(f"[db-removed] {counts}")
    finally:
        conn.close()


def main():
    global DB_PATH_OVERRIDE
    ap = argparse.ArgumentParser(description="Delete stored documents and/or vector records.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-id", action="append", help="doc_id to delete (repeatable)", dest="doc_ids")
    g.add_argument("--all", action="store_true", help="delete all")
    ap.add_argument("--vectors-only", action="store_true", help="delete only from document_vectors (no file or workspace_documents removal)")
    ap.add_argument("--dry-run", action="store_true", help="preview without deleting")
    ap.add_argument("--db-path", help="override sqlite db path (absolute or relative to CWD)")
    args = ap.parse_args()

    if args.db_path:
        p = Path(args.db_path)
        DB_PATH_OVERRIDE = p if p.is_absolute() else (Path.cwd() / p).resolve()
        print(f"[db] using override: {DB_PATH_OVERRIDE}")

    if args.doc_ids:
        delete_by_doc_ids(args.doc_ids, dry_run=args.dry_run, vectors_only=args.vectors_only)
    elif args.all:
        delete_all(dry_run=args.dry_run, vectors_only=args.vectors_only)


if __name__ == "__main__":
    main()


"""
특정 문서만(미리보기): 
    python scripts/delete_workspace_documents.py --doc-id b3f9cf63-... --dry-run
특정 문서 벡터만 삭제: 
    python scripts/delete_workspace_documents.py --doc-id DOC_ID --vectors-only
모든 벡터 삭제(미리보기): 
    python scripts/delete_workspace_documents.py --all --vectors-only
전체 미리보기: 
    python scripts/delete_workspace_documents.py --all --dry-run
전체 문서 + 백터 삭제: 
    python scripts/delete_workspace_documents.py --all

python scripts/delete_workspace_documents.py --all --vectors-only --db-path backend/storage/pps_rag.db
"""