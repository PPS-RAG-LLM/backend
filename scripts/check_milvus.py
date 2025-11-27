
import sys
import os
# 프로젝트 루트를 path에 추가
sys.path.append(os.getcwd())

from service.vector_db.milvus_store import get_milvus_client, resolve_collection, run_dense_search, run_hybrid_search
from storage.db_models import DocumentType
from pymilvus import AnnSearchRequest, RRFRanker

def check_milvus_data(doc_id):
    client = get_milvus_client()
    collection_name = resolve_collection(DocumentType.TEMP.value)
    
    print(f"Checking collection: {collection_name}")
    
    # 컬렉션 존재 확인
    if collection_name not in client.list_collections():
        print(f"Collection {collection_name} does not exist.")
        return

    # 1. 데이터 존재 확인 (Query)
    print("\n--- 1. Query Check ---")
    res = client.query(
        collection_name=collection_name,
        filter=f'doc_id == "{doc_id}"',
        output_fields=["doc_id", "workspace_id", "text"],
        limit=3
    )
    print(f"Found {len(res)} items.")
    if len(res) > 0:
        print(f"First item sample: {res[0]}")
    else:
        print("No items found by query. Exiting.")
        return

    # 임시 벡터 생성 (0 벡터) - 실제로는 임베딩 필요하지만 존재 여부 체크용
    dim = 1024 # Qwen3-14B? 아니 임베딩 모델에 따라 다름. 로그에서 1024나 768 등 확인 필요.
    # 로그의 QUERY_VEC 길이를 보면 꽤 깁니다. 로그 샘플은 잘려서 보이지만..
    # 대략 1024라고 가정하고 시도. 에러나면 차원 맞춤.
    # 로그에서: -0.415... 등등.
    # 임시로 차원 확인
    coll_info = client.describe_collection(collection_name)
    # dim 확인
    for field in coll_info['fields']:
        if field['name'] == 'embedding':
            dim = field['params']['dim']
            print(f"Detected embedding dim: {dim}")
            break
    
    dummy_vec = [0.1] * int(dim)
    
    # 2. Dense Search Test
    print("\n--- 2. Dense Search Test ---")
    filter_expr = f'doc_id == "{doc_id}"'
    try:
        dense_res = client.search(
            collection_name=collection_name,
            data=[dummy_vec],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=10,
            filter=filter_expr,
            output_fields=["doc_id"]
        )
        print(f"Dense search hits: {len(dense_res[0])}")
    except Exception as e:
        print(f"Dense search failed: {e}")

    # 3. Hybrid Search Test
    print("\n--- 3. Hybrid Search Test ---")
    
    filter_expr_in = f'doc_id in ["{doc_id}"]'
    
    # 실제 쿼리 벡터 생성 필요하지만 여기선 dummy_vec 사용하므로 벡터 점수는 무의미할 수 있음.
    # 하지만 BM25 점수도 포함됨.
    # 쿼리 텍스트 "ㅎㅇ"
    query_text = "ㅎㅇ"
    
    try:
        dense_req = AnnSearchRequest(
            data=[dummy_vec], # 실제로는 의미 없는 벡터라 점수 낮음
            anns_field="embedding",
            param={"metric_type": "IP", "params": {}},
            limit=10
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="text_sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=10
        )
        
        hybrid_res = client.hybrid_search(
            collection_name=collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=10,
            filter=filter_expr_in, 
            output_fields=["doc_id"]
        )
        print(f"Hybrid search hits: {len(hybrid_res[0])}")
        for i, hit in enumerate(hybrid_res[0]):
            print(f"Hit {i}: score={hit['score']}, doc_id={hit['entity']['doc_id']}")

    except Exception as e:
        print(f"Hybrid search failed: {e}")

if __name__ == "__main__":
    # 로그에 나온 docId 사용
    target_doc_id = "7c91c3e8-6f33-4447-8582-1a902d87acae"
    check_milvus_data(target_doc_id)

