
import sys
import os
sys.path.append(os.getcwd())

from utils.database import get_session
from storage.db_models import DocumentVector
from sqlalchemy import select

def check_db(doc_id):
    with get_session() as session:
        stmt = select(DocumentVector).where(DocumentVector.doc_id == doc_id)
        rows = session.execute(stmt).scalars().all()
        print(f"Found {len(rows)} vectors in DB.")
        if rows:
            print(f"Embedding Version: {rows[0].embedding_version}")
            print(f"Collection: {rows[0].collection}")

if __name__ == "__main__":
    target_doc_id = "7c91c3e8-6f33-4447-8582-1a902d87acae"
    check_db(target_doc_id)

