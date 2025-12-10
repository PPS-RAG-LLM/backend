"""Summary용 문서 전체 로드"""
from typing import Dict, Any, List
from pathlib import Path
from config import config
from repository.documents import list_workspace_documents
from utils import logger
import json

logger = logger(__name__)

FULL_TEXT_DIR = Path(config.get("full_text_dir", "storage/documents/full_text"))


def get_full_documents_texts(workspace_id: int) -> List[Dict[str, Any]]:
    """
   Summary/DocGen용으로 워크스페이스 문서를 로드 (from storage/documents/full_text/)
    """
    
    try:
        # 1. 워크스페이스에 등록된 문서 목록 조회
        workspace_docs = list_workspace_documents(workspace_id)
        if not workspace_docs:
            logger.info("워크스페이스에 등록된 문서가 없습니다")
            return []
        
        full_documents = []

        for doc in workspace_docs:
            doc_id      = doc.get("doc_id")
            filename    = doc.get("filename")
            
            if not doc_id:
                logger.warning(f"docpath 없음: {doc_id}")
                continue

            # 2. .txt 파일 경로 확인
            text_file = FULL_TEXT_DIR / f"{doc_id}.txt"

            full_text = ""
            if text_file.exists():
                try:
                    full_text = text_file.read_text(encoding="utf-8")
                except Exception as e:
                    logger.error(f"문서 로드 실패 {doc_id}: {e}")
            else:
                # 파일이 없으면 (마이그레이션 전 or 업로드 실패)
                # 로그만 남기고 스킵
                logger.warning(f"❗️❗️❗️ Full text file not found: {text_file}")
                continue

            if full_text:
                full_documents.append({
                    "doc_id"    : doc_id, 
                    "title"     : filename,
                    "text"      : full_text,
                    "word_count": len(full_text.split())    # word_count 등은 계산하거나 메타데이터에서 가져옴
                })
                logger.info(f"문서 로드 : {filename} ({len(full_text)} chars)")
        
        logger.info(f"## 최종 로드된 문서 수: {len(full_documents)}")
        return full_documents
        
    except Exception as e:
        logger.error(f"Full document load failed: {e}")
        return []

