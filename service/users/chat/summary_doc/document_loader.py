"""Summary용 문서 전체 로드"""
from typing import Dict, Any, List
from pathlib import Path
from utils import logger
import json

logger = logger(__name__)


def get_full_documents_for_summary(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Summary용으로 첨부 문서의 전체 텍스트를 가져옴 (벡터화 X, 파싱된 텍스트만)
    
    특징:
    - 벡터 검색 없이 전체 문서 텍스트를 로드
    - documents-info의 pageContent 사용
    - 요약 작업에 적합
    
    Returns:
        List of documents with full text from pageContent
    """
    from ..retrieval import extract_doc_ids_from_attachments
    
    try:
        # 1. attachments에서 doc_id 추출
        temp_doc_ids = extract_doc_ids_from_attachments(body.get("attachments"))
        if not temp_doc_ids:
            return []
        
        logger.info(f"\n## Summary용 문서 로드: {temp_doc_ids}\n")
        
        # 2. documents-info에서 pageContent 읽기
        doc_info_dir = Path("storage/documents/documents-info")
        
        full_documents = []
        for doc_id in temp_doc_ids:
            # 문서 정보 파일 찾기
            for info_file in doc_info_dir.glob(f"*{doc_id}.json"):
                try:
                    doc_info = json.loads(info_file.read_text(encoding="utf-8"))
                    
                    # pageContent에서 전체 텍스트 가져오기
                    full_text = doc_info.get("pageContent", "")
                    
                    if full_text:
                        full_documents.append({
                            "doc_id": doc_id,
                            "title": doc_info.get("title", "Unknown"),
                            "text": full_text,
                            "word_count": doc_info.get("wordCount", 0),
                            "token_estimate": doc_info.get("token_count_estimate", 0)
                        })
                        
                        logger.info(f"✅ 문서 로드: {doc_info.get('title')}, 토큰: {doc_info.get('token_count_estimate', 0)}")
                    break
                    
                except Exception as e:
                    logger.error(f"문서 로드 실패 {doc_id}: {e}")
        
        return full_documents
        
    except Exception as e:
        logger.error(f"Full document load failed: {e}")
        return []

