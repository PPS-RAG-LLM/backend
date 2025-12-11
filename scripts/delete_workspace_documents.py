# scripts/delete_workspace_documents.py
import argparse
import sys
from pathlib import Path
from typing import List, Dict
# 프로젝트 루트 추가 (backend/에서 실행하지 않아도 import 가능)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402
from utils.database import get_session
from sqlalchemy import select, delete, func, and_
from storage.db_models import Document, DocumentVector, DocumentType

WORKSPACE_DOC_TYPE = "WS_DOCS"

# 파일 저장 위치(존재하지 않을 수도 있어 기본값 준비)
SAVE_DOC_DIR = config.get("user_documents", {})
DOC_INFO_DIR = Path(SAVE_DOC_DIR.get("doc_info_dir", "storage/documents/documents-info"))
VEC_CACHE_DIR = Path(SAVE_DOC_DIR.get("vector_cache_dir", "storage/documents/vector-cache"))

def _fetch_docs_by_ids(doc_ids: List[str]) -> Dict[str, dict]:
    if not doc_ids:
        return {}
    with get_session() as session:
        stmt = select(Document.doc_id, Document.filename, Document.source_path).where(
            and_(
                Document.doc_type == WORKSPACE_DOC_TYPE,
                Document.doc_id.in_(doc_ids)
            )
        )
        rows = session.execute(stmt).all()
        # dict 형태로 변환
        return {
            row.doc_id: {"doc_id": row.doc_id, "filename": row.filename, "docpath": row.source_path} 
            for row in rows
        }

def _fetch_all_docs() -> List[dict]:
    with get_session() as session:
        stmt = select(Document.doc_id, Document.filename, Document.source_path).where(
            Document.doc_type == WORKSPACE_DOC_TYPE
        ).order_by(Document.id.desc())
        rows = session.execute(stmt).all()
        return [{"doc_id": row.doc_id, "filename": row.filename, "docpath": row.source_path} for row in rows]

def _count_vectors() -> int:
    with get_session() as session:
        stmt = select(func.count(DocumentVector.id))
        return session.execute(stmt).scalar() or 0

def _count_vectors_for(doc_ids: List[str]) -> int:
    if not doc_ids:
        return 0
    with get_session() as session:
        stmt = select(func.count(DocumentVector.id)).where(DocumentVector.doc_id.in_(doc_ids))
        return session.execute(stmt).scalar() or 0

def _delete_vectors_by_doc_ids(doc_ids: List[str]) -> int:
    if not doc_ids:
        return 0
    before = _count_vectors_for(doc_ids)
    
    with get_session() as session:
        stmt = delete(DocumentVector).where(DocumentVector.doc_id.in_(doc_ids))
        result = session.execute(stmt)
        session.commit()
        deleted = int(result.rowcount or 0)
        
    after = _count_vectors_for(doc_ids)
    print(f"[vectors] before={before}, deleted={deleted}, after={after}")
    return deleted

def _delete_all_vectors() -> int:
    before = _count_vectors()
    with get_session() as session:
        stmt = delete(DocumentVector)
        result = session.execute(stmt)
        session.commit()
        deleted = int(result.rowcount or 0)
    
    after = _count_vectors()
    print(f"[vectors] before={before}, deleted={deleted}, after={after}")
    return deleted

def _delete_db_records(doc_ids: List[str]) -> Dict[str, int]:
    if not doc_ids:
        return {"documents": 0, "document_vectors": 0}
    
    with get_session() as session:
        # 1) document_vectors 삭제 (명시적 삭제)
        # (Postgres FK CASCADE가 걸려있다면 자동 삭제되지만, 여기서는 명시적으로 카운트 및 삭제)
        del_vec_stmt = delete(DocumentVector).where(DocumentVector.doc_id.in_(doc_ids))
        result_vec = session.execute(del_vec_stmt)
        dv = int(result_vec.rowcount or 0)
        
        # 2) documents 삭제
        del_doc_stmt = delete(Document).where(
            and_(
                Document.doc_type == WORKSPACE_DOC_TYPE,
                Document.doc_id.in_(doc_ids)
            )
        )
        result_doc = session.execute(del_doc_stmt)
        wd = int(result_doc.rowcount or 0)
        
        session.commit()
        
    return {"documents": wd, "document_vectors": dv}

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
    if vectors_only:
        if dry_run:
            print(f"[dry-run] would delete document_vectors for doc_ids: {doc_ids}")
        else:
            _delete_vectors_by_doc_ids(doc_ids)
        return

    rows = _fetch_docs_by_ids(doc_ids)

    # 파일 삭제(문서 메타가 있는 것만)
    if rows:
        for doc_id in doc_ids:
            row = rows.get(doc_id)
            if not row:
                print(f"[skip] doc_id not found in documents table: {doc_id}")
                continue
            _remove_files(
                filename=row["filename"],
                doc_id=row["doc_id"],
                doc_rel_path=row["docpath"] or "",
                dry_run=dry_run,
            )
    else:
        print("[info] no matching workspace documents; will still delete document_vectors for provided doc_ids.")

    # DB 삭제
    if dry_run:
        print(f"[dry-run] would delete DB rows(documents + document_vectors) for: {doc_ids}")
    else:
        counts = _delete_db_records(doc_ids)
        print(f"[db-removed] {counts}")

def delete_all(dry_run: bool = True, vectors_only: bool = False) -> None:
    if vectors_only:
        if dry_run:
            total = _count_vectors()
            print(f"[dry-run] would delete ALL rows from document_vectors (current={total})")
        else:
            _delete_all_vectors()
        return

    rows = _fetch_all_docs()
    if not rows:
        print("[info] workspace documents are empty.")
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
        print(f"[dry-run] would delete ALL DB rows(documents + document_vectors): total={len(doc_ids)}")
    else:
        counts = _delete_db_records(doc_ids)
        print(f"[db-removed] {counts}")

def main():
    ap = argparse.ArgumentParser(description="Delete stored documents and/or vector records.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--doc-id", action="append", help="doc_id to delete (repeatable)", dest="doc_ids")
    g.add_argument("--all", action="store_true", help="delete all")
    ap.add_argument("--vectors-only", action="store_true", help="delete only from document_vectors (no file or documents removal)")
    ap.add_argument("--dry-run", action="store_true", help="preview without deleting")
    # --db-path 옵션은 이제 무시됨 (utils.database 설정 따름)
    ap.add_argument("--db-path", help="[DEPRECATED] override sqlite db path (ignored in PostgreSQL mode)")
    args = ap.parse_args()

    if args.db_path:
        print("[warn] --db-path argument is deprecated and ignored. Using config.yaml database settings.")

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
"""
