"""
Chat 서비스 모듈

각 카테고리별로 명확하게 구분된 구조:
- qna/: QA 카테고리 (RAG 검색 + Chat history)
- summary/: Summary 카테고리 (전체 문서 요약)
- doc_gen/: Doc Gen 카테고리 (템플릿 기반 문서 생성)
- common/: 공통 로직 (검증, 메시지 구성, 스트리밍)
- retrieval/: RAG 검색 로직
"""
from .qna import stream_chat_for_qna
from .summary import stream_chat_for_summary
from .doc_gen import stream_chat_for_doc_gen

__all__ = [
    "stream_chat_for_qna",
    "stream_chat_for_summary",
    "stream_chat_for_doc_gen",
]
