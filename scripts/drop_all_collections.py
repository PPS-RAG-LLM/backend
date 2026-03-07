import sys
import os

# 프로젝트 루트를 path에 추가 (스크립트 위치 기준 상위 디렉토리)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.vector_db.milvus_store import get_milvus_client, drop_all_collections

if __name__ == "__main__":
    print("Connecting to Milvus...")
    try:
        client = get_milvus_client()
        print("Dropping all collections...")
        deleted_cols = drop_all_collections(client)
        print(f"Deleted collections: {deleted_cols}")
        print("Done.")
    except Exception as e:
        print(f"Error: {e}")
