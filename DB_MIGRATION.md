# DB Migration & Constraints Guide

## 📋 개요

SQLite 원시 SQL에서 SQLAlchemy ORM으로 전환하면서 **데이터베이스 레벨에서 처리되던 제약사항들을 애플리케이션 레벨에서 구현**해야 합니다.

---

## 🔥 **1. 데이터베이스 트리거 → 서비스 로직 이전**

### 1.1 **시스템 프롬프트 템플릿 기본값 관리**

**기존 트리거 (schema.sql)**:

```sql
-- 새 기본 템플릿 삽입 시 기존 기본값 자동 해제
CREATE TRIGGER trg_spt_before_insert_single_default
BEFORE INSERT ON system_prompt_template
WHEN NEW.is_default = 1
BEGIN
  UPDATE system_prompt_template SET is_default = 0 WHERE category = NEW.category;
END;
```

**→ 서비스 로직 구현 필요**:

- **위치**: `service/admin/manage_admin_LLM.py` 또는 새로운 `service/admin/prompt_template.py`
- **함수**: `set_default_prompt_template(template_id: int, category: str)`

```python
# service/admin/prompt_template.py
from utils.database import get_session
from storage.db_models import SystemPromptTemplate

def set_default_prompt_template(template_id: int, category: str) -> bool:
    """
    카테고리별 기본 프롬프트 템플릿 설정
    - 새 기본값 설정 시 동일 카테고리의 기존 기본값 자동 해제
    """
    with get_session() as session:
        # 1. 기존 기본값 해제
        session.query(SystemPromptTemplate).filter(
            SystemPromptTemplate.category == category,
            SystemPromptTemplate.is_default == True
        ).update({"is_default": False})

        # 2. 새 기본값 설정
        session.query(SystemPromptTemplate).filter(
            SystemPromptTemplate.id == template_id
        ).update({"is_default": True})

        session.commit()
        return True
```

### 1.2 **LLM 모델 기본값 관리**

**기존 트리거 (schema.sql)**:

```sql
-- 새 기본 모델 삽입 시 기존 기본값 자동 해제
CREATE TRIGGER trg_llm_before_insert_single_default
BEFORE INSERT ON llm_models
WHEN NEW.is_default = 1
BEGIN
  UPDATE llm_models SET is_default = 0 WHERE category = NEW.category;
END;
```

**→ 서비스 로직 구현 필요**:

- **위치**: `service/admin/manage_admin_LLM.py`
- **함수**: `set_default_llm_model(model_id: int, category: str)`

```python
# service/admin/manage_admin_LLM.py에 추가
from storage.db_models import LlmModel

def set_default_llm_model(model_id: int, category: str) -> bool:
    """
    카테고리별 기본 LLM 모델 설정
    - 새 기본값 설정 시 동일 카테고리의 기존 기본값 자동 해제
    """
    with get_session() as session:
        # 1. 기존 기본값 해제
        session.query(LlmModel).filter(
            LlmModel.category == category,
            LlmModel.is_default == True
        ).update({"is_default": False})

        # 2. 새 기본값 설정
        session.query(LlmModel).filter(
            LlmModel.id == model_id
        ).update({"is_default": True})

        session.commit()
        return True
```

---

## 🔒 **2. 싱글톤 테이블 제약조건**

### 2.1 **벡터 설정 (vector_settings)**

**기존 제약조건 (schema.sql)**:

```sql
CREATE TABLE vector_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- 오직 id=1만 허용
    ...
);
```

**→ 서비스 로직 구현 필요**:

- **위치**: `service/users/workspace.py` 또는 새로운 `service/admin/system_settings.py`

```python
# service/admin/system_settings.py
from storage.db_models import VectorSettings, RagSettings
from utils.time import now_kst_string

def update_vector_settings(search_type: str, chunk_size: int, overlap: int) -> bool:
    """
    벡터 검색 설정 업데이트 (싱글톤 테이블)
    - id=1 레코드만 존재하도록 강제
    """
    with get_session() as session:
        settings = session.query(VectorSettings).filter(VectorSettings.id == 1).first()

        if settings:
            # 기존 레코드 업데이트
            settings.search_type = search_type
            settings.chunk_size = chunk_size
            settings.overlap = overlap
            settings.updated_at = now_kst_string()
        else:
            # 새 레코드 생성 (id=1 강제)
            settings = VectorSettings(
                id=1,
                search_type=search_type,
                chunk_size=chunk_size,
                overlap=overlap
            )
            session.add(settings)

        session.commit()
        return True

def get_vector_settings() -> VectorSettings:
    """벡터 설정 조회 (싱글톤)"""
    with get_session() as session:
        return session.query(VectorSettings).filter(VectorSettings.id == 1).first()
```

### 2.2 **RAG 전역 설정 (rag_settings)**

```python
def update_rag_settings(search_type: str, chunk_size: int, overlap: int, embedding_key: str) -> bool:
    """
    RAG 전역 설정 업데이트 (싱글톤 테이블)
    - id=1 레코드만 존재하도록 강제
    """
    with get_session() as session:
        settings = session.query(RagSettings).filter(RagSettings.id == 1).first()

        if settings:
            settings.search_type = search_type
            settings.chunk_size = chunk_size
            settings.overlap = overlap
            settings.embedding_key = embedding_key
            settings.updated_at = now_kst_string()
        else:
            settings = RagSettings(
                id=1,
                search_type=search_type,
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_key=embedding_key
            )
            session.add(settings)

        session.commit()
        return True
```

---

## 🔍 **3. 조건부 유니크 인덱스 → 서비스 검증**

### 3.1 **임베딩 모델 활성 상태 관리**

**기존 인덱스 (schema.sql)**:

```sql
-- "활성 = 1" 인 레코드는 최대 1개만
CREATE UNIQUE INDEX ux_embedding_models_active_one ON embedding_models (is_active)
WHERE is_active = 1;
```

**→ 서비스 로직 구현 필요**:

- **위치**: `service/admin/manage_admin_LLM.py`

```python
# service/admin/manage_admin_LLM.py에 추가
from storage.db_models import EmbeddingModel

def activate_embedding_model(model_id: int) -> bool:
    """
    임베딩 모델 활성화
    - 기존 활성 모델 자동 비활성화 (하나만 활성 허용)
    """
    with get_session() as session:
        # 1. 모든 기존 모델 비활성화
        session.query(EmbeddingModel).filter(
            EmbeddingModel.is_active == 1
        ).update({"is_active": 0, "activated_at": None})

        # 2. 새 모델 활성화
        session.query(EmbeddingModel).filter(
            EmbeddingModel.id == model_id
        ).update({
            "is_active": 1,
            "activated_at": now_kst_string()
        })

        session.commit()
        return True

def get_active_embedding_model() -> EmbeddingModel:
    """현재 활성 임베딩 모델 조회"""
    with get_session() as session:
        return session.query(EmbeddingModel).filter(
            EmbeddingModel.is_active == 1
        ).first()
```

---

## ✅ **4. CHECK 제약조건 → 서비스 검증**

### 4.1 **열거형 값 검증**

**기존 제약조건들 (schema.sql)**:

```sql
-- 워크스페이스 카테고리
CHECK ("category" IN ('qa', 'doc_gen', 'summary'))

-- LLM 모델 타입
CHECK ("type" IN ('base', 'lora', 'full'))

-- 채팅 피드백 값
CHECK ("value" IN (1, -1))
```

**→ 서비스 로직 구현 필요**:

- **위치**: `utils/validators.py` (신규 생성)

```python
# utils/validators.py
from enum import Enum
from typing import Any

class WorkspaceCategory(Enum):
    QA = "qa"
    DOC_GEN = "doc_gen"
    SUMMARY = "summary"

class LlmModelType(Enum):
    BASE = "base"
    LORA = "lora"
    FULL = "full"

class ChatFeedbackValue(Enum):
    POSITIVE = 1
    NEGATIVE = -1

def validate_workspace_category(category: str) -> bool:
    """워크스페이스 카테고리 검증"""
    return category in [c.value for c in WorkspaceCategory]

def validate_llm_model_type(model_type: str) -> bool:
    """LLM 모델 타입 검증"""
    return model_type in [t.value for t in LlmModelType]

def validate_chat_feedback_value(value: int) -> bool:
    """채팅 피드백 값 검증"""
    return value in [v.value for v in ChatFeedbackValue]

# 범용 검증 함수
def validate_positive_integer(value: int, field_name: str = "value") -> bool:
    """양수 검증 (chunk_size, top_n 등)"""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return True

def validate_non_negative_integer(value: int, field_name: str = "value") -> bool:
    """음이 아닌 정수 검증 (overlap 등)"""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return True
```

### 4.2 **서비스 레이어에서 검증 적용**

```python
# service/users/workspace.py에 추가
from utils.validators import validate_workspace_category, validate_positive_integer

def create_workspace(name: str, category: str, top_n: int = 4, **kwargs) -> int:
    """워크스페이스 생성 (검증 포함)"""
    # 입력값 검증
    if not validate_workspace_category(category):
        raise ValueError(f"Invalid category: {category}")

    validate_positive_integer(top_n, "top_n")

    with get_session() as session:
        workspace = Workspace(
            name=name,
            category=category,
            top_n=top_n,
            **kwargs
        )
        session.add(workspace)
        session.commit()
        session.refresh(workspace)
        return workspace.id
```

---

## 🚀 **5. 구현 우선순위**

### **Phase 1: 필수 제약조건** (즉시 구현)

1. **LLM 모델 기본값 관리** - `service/admin/manage_admin_LLM.py`
2. **임베딩 모델 활성화** - `service/admin/manage_admin_LLM.py`
3. **입력값 검증** - `utils/validators.py` 신규 생성

### **Phase 2: 시스템 설정** (다음 단계)

1. **프롬프트 템플릿 관리** - `service/admin/prompt_template.py` 신규
2. **시스템 설정 관리** - `service/admin/system_settings.py` 신규

### **Phase 3: 최적화** (선택적)

1. **성능 모니터링** - 트리거 대신 애플리케이션 로직 성능 확인
2. **데이터 일관성 검증** - 배치 작업으로 정기 검증

---

## 📝 **6. 마이그레이션 체크리스트**

### **기존 코드 수정**

- [ ] `repository/users/llm_models.py` - 기본값 설정 로직 추가
- [ ] `service/admin/manage_admin_LLM.py` - 제약조건 검증 추가
- [ ] `service/users/workspace.py` - 입력값 검증 추가

### **신규 파일 생성**

- [ ] `utils/validators.py` - 검증 함수들
- [ ] `service/admin/prompt_template.py` - 프롬프트 템플릿 관리
- [ ] `service/admin/system_settings.py` - 시스템 설정 관리

### **테스트**

- [ ] 기본값 자동 전환 테스트
- [ ] 싱글톤 테이블 제약조건 테스트
- [ ] 입력값 검증 테스트
- [ ] 기존 기능 회귀 테스트

---

## ⚠️ **주의사항**

1. **트랜잭션 관리**: 기본값 전환 로직은 반드시 트랜잭션 내에서 실행
2. **동시성**: 여러 사용자가 동시에 기본값 설정 시 경쟁 조건 고려
3. **데이터 일관성**: 기존 데이터베이스의 제약조건 위반 여부 사전 확인
4. **성능**: 트리거 대신 애플리케이션 로직 사용으로 인한 성능 영향 모니터링

---

## 🔗 **관련 파일**

- **모델 정의**: `storage/models.py`
- **DB 설정**: `utils/database.py`
- **스키마**: `storage/schema.sql`
- **기존 서비스**: `service/admin/manage_admin_LLM.py`
- **기존 서비스**: `service/users/workspace.py`
