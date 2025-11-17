# Chat 서비스 모듈 구조

각 카테고리별로 명확하게 구분된 채팅 서비스 모듈입니다.

## 디렉토리 구조

```
service/users/chat/
├── __init__.py                    # 통합 export
│
├── common/                        # 📦 공통 모듈
│   ├── __init__.py
│   ├── validators.py              # Preflight 검증
│   ├── message_builder.py         # 메시지 구성 함수
│   └── stream_handler.py          # 스트리밍 및 DB 저장
│
├── qna/                            # ✅ QA 카테고리
│   ├── __init__.py
│   └── qna.py                      # QA 스트림 로직
│
├── summary/                       # ✅ Summary 카테고리
│   ├── __init__.py
│   ├── summary.py                 # Summary 스트림 로직
│   └── document_loader.py         # 문서 전체 로드
│
├── doc_gen/                       # ✅ Doc Gen 카테고리
│   ├── __init__.py
│   └── doc_gen.py                 # Doc Gen 스트림 로직
│
└── retrieval/                     # 🔍 RAG 검색
    ├── __init__.py
    └── chat_retrieval.py          # RAG 검색 로직
```

---

## 카테고리별 책임

### 1. **QA (Q&A 대화)**
**위치**: `qna/qna.py`

**특징**:
- RAG 검색으로 관련 문서 청크 검색
- Chat history 포함 (이전 대화 기록 활용)
- 사용자 질문에 대한 답변 생성

**흐름**:
1. Preflight 검증 → 워크스페이스 및 스레드 확인
2. Chat history 로드 → 이전 대화 기록 가져오기
3. RAG context 검색 → 워크스페이스 문서 + 첨부 문서에서 관련 청크 검색
4. User message 구성 → RAG context를 포함한 사용자 메시지
5. 스트리밍 및 저장 → LLM 응답 생성 및 DB 저장

---

### 2. **Summary (문서 요약)**
**위치**: `summary/summary.py`, `summary/document_loader.py`

**특징**:
- 전체 문서 로드 (벡터화 없이 전체 텍스트)
- `originalText` 또는 `attachments` 중 최소 하나 필수
- 둘 다 있으면 모두 CONTEXTS로 포함
- 추가 요청사항(`userPrompt`) 지원

**흐름**:
1. Preflight 검증 → 워크스페이스 확인
2. 문서 전체 로드 → `documents-info`의 `pageContent` 사용
3. 메시지 구성 → `originalText` + `parsed_documents` 결합
4. User message 구성 → 전체 문서 내용 포함
5. 스트리밍 및 저장 → LLM 응답 생성 및 DB 저장

**주의**:
- Summary는 RAG 검색을 **사용하지 않음**
- 문서를 청크로 나누지 않고 **전체 텍스트**를 사용

---

### 3. **Doc Gen (문서 생성)**
**위치**: `doc_gen/doc_gen.py`

**특징**:
- 템플릿 기반 문서 생성
- 변수 치환 (`templateVariables`)
- 양식 생성 및 포맷팅

**흐름**:
1. Preflight 검증 → 워크스페이스 확인
2. 템플릿 렌더링 → 변수 치환 및 시스템 프롬프트 구성
3. RAG context (선택적) → 필요시 관련 문서 검색
4. User message 구성 → 템플릿 변수 포함
5. 스트리밍 및 저장 → LLM 응답 생성 및 DB 저장

---

## 공통 모듈 (`common/`)

### 1. **validators.py**
- `preflight_stream_chat_for_workspace()`: 스트리밍 전 검증
  - 카테고리 검증
  - 워크스페이스 존재 확인
  - 스레드 존재 확인 (QA만)
  - 모드 검증 (chat/query)

### 2. **message_builder.py**
- `build_system_message()`: 시스템 프롬프트 구성
- `build_user_message_with_context()`: RAG context를 포함한 사용자 메시지
- `render_template()`: 템플릿 렌더링 (Doc Gen용)
- `resolve_runner()`: LLM streamer 생성

### 3. **stream_handler.py**
- `stream_and_persist()`: 스트리밍 응답 생성 및 DB 저장
  - LLM 스트리밍
  - `sources` 필드 구성 (RAG 메타데이터)
  - 응답 JSON 저장
  - 임시 벡터 정리

---

## 사용 예시

### Router에서 사용

```python
from service.users.chat import (
    stream_chat_for_qna,
    stream_chat_for_summary,
    stream_chat_for_doc_gen,
)

# QA
gen = stream_chat_for_qna(
    user_id=user_id,
    slug=slug,
    thread_slug=thread_slug,
    category="qna",
    body=body.model_dump(),
)

# Summary
gen = stream_chat_for_summary(
    user_id=user_id,
    slug=slug,
    category="summary",
    body=body_dict,
)

# Doc Gen
gen = stream_chat_for_doc_gen(
    user_id=user_id,
    slug=slug,
    category="doc_gen",
    body=body_dict,
)
```

---

## 주요 개선 사항

### ✅ **명확한 책임 분리**
- 각 카테고리가 독립된 디렉토리/모듈
- 공통 로직은 `common/`에 집중
- 검색 로직은 `retrieval/`에 집중

### ✅ **유지보수성 향상**
- 카테고리별로 코드 수정 범위 제한
- 공통 함수 재사용으로 중복 제거
- 테스트 작성이 쉬워짐

### ✅ **확장성**
- 새로운 카테고리 추가가 간단함
- 각 카테고리가 독립적으로 발전 가능
- 공통 로직 변경 시 영향 범위 최소화

---

## 주의사항

1. **QA와 Summary의 차이**
   - QA: RAG 검색 사용, 관련 청크만 검색
   - Summary: RAG 사용 안 함, 전체 문서 로드

2. **sources 필드**
   - QA: RAG 검색 결과 (doc_id, title, page, chunk_index, score)
   - Summary: 전체 문서 정보 (doc_id, title, text)
   - Doc Gen: RAG 검색 결과 (선택적)

3. **임시 문서 정리**
   - 모든 카테고리에서 `temp_doc_ids` 자동 삭제
   - 스트리밍 완료 후 벡터 정리

---

## 마이그레이션 노트

기존 `chat.py`에서 다음과 같이 분리되었습니다:

| 기존 함수                        | 새 위치                              |
|--------------------------------|-------------------------------------|
| `preflight_stream_chat_for_workspace` | `common/validators.py`              |
| `_build_system_message`         | `common/message_builder.py`         |
| `_build_user_message_with_context` | `common/message_builder.py`      |
| `_render_template`              | `common/message_builder.py`         |
| `_resolve_runner`               | `common/message_builder.py`         |
| `_stream_and_persist`           | `common/stream_handler.py`          |
| `stream_chat_for_qna`            | `qna/qna.py`                          |
| `insert_rag_context` (QA)       | `qna/qna.py` (private)                |
| `stream_chat_for_summary`       | `summary/summary.py`                |
| `_compose_summary_message`      | `summary/summary.py` (private)      |
| `get_full_documents_for_summary`| `summary/document_loader.py`        |
| `stream_chat_for_doc_gen`       | `doc_gen/doc_gen.py`                |
| `_compose_doc_gen_message`      | `doc_gen/doc_gen.py` (private)      |

---

## 추가 개발 가이드

### 새 카테고리 추가 시

1. `service/users/chat/{category}/` 디렉토리 생성
2. `__init__.py` 및 `{category}.py` 작성
3. `service/users/chat/__init__.py`에 export 추가
4. Router에서 import 및 endpoint 추가

### 공통 로직 수정 시

- `common/` 모듈만 수정하면 모든 카테고리에 적용됨
- 단, 각 카테고리의 특수한 로직은 해당 카테고리 내부에서만 수정

---

**작성일**: 2025-10-02  
**버전**: 1.0  
**작성자**: AI Assistant

