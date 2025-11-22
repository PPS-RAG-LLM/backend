"""Summary용 문서 전체 로드"""
from typing import Dict, Any, List
from pathlib import Path
from repository.documents import list_workspace_documents
from utils import logger
import json

logger = logger(__name__)


def get_full_documents_texts(workspace_id: int) -> List[Dict[str, Any]]:
    """
   Summary용으로 워크스페이스에 등록된 문서의 전체 텍스트를 가져옴
    
    Args:
        workspace_id: 워크스페이스 ID
        attachments: 선택할 문서 정보 (name 필드로 필터링, 없으면 전체)
    
    Returns:
        List of documents with full text from pageContent
    """
    
    try:
        # 1. 워크스페이스에 등록된 문서 목록 조회
        workspace_docs = list_workspace_documents(workspace_id)
        if not workspace_docs:
            logger.info("워크스페이스에 등록된 문서가 없습니다")
            return []
        
        # 2. documents-info에서 pageContent 읽기
        doc_info_dir = Path("storage/documents/documents-info")
        full_documents = []

        for doc in workspace_docs:
            filename = doc.get("filename", "")
            
            doc_id = doc.get("doc_id")
            docpath = doc.get("docpath")
            
            if not docpath:
                logger.warning(f"docpath 없음: {filename}")
                continue
                
            # docpath에서 파일 읽기
            doc_file = Path(docpath)
            if not doc_file.exists():
                # 상대 경로로 재시도
                doc_file = doc_info_dir / Path(docpath).name
                
            if doc_file.exists():
                try:
                    doc_info = json.loads(doc_file.read_text(encoding="utf-8"))
                    full_text = doc_info.get("pageContent", "")
                    
                    if full_text:
                        full_documents.append({
                            "doc_id": doc_id,
                            "title": doc_info.get("title", filename),
                            "text": full_text,
                            "word_count": doc_info.get("wordCount", 0),
                            "token_estimate": doc_info.get("token_count_estimate", 0)
                        })
                        
                        logger.info(f"✅ 문서 로드: {doc_info.get('title')}, 토큰: {doc_info.get('token_count_estimate', 0)}")
                    else:
                        logger.warning(f"pageContent 없음: {filename}")
                        
                except Exception as e:
                    logger.error(f"문서 로드 실패 {doc_id}: {e}")
            else:
                logger.warning(f"파일 없음: {docpath}")
        
        logger.info(f"## 최종 로드된 문서 수: {len(full_documents)}")
        return full_documents
        
    except Exception as e:
        logger.error(f"Full document load failed: {e}")
        return []