from __future__ import annotations

from sqlalchemy import Column, Integer, Text, DateTime, Boolean, ForeignKey, text, UniqueConstraint, Float, Index
from sqlalchemy.orm import relationship
from utils.database import Base

class WorkspaceDocument(Base):
    __tablename__ = "workspace_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Text, nullable=False, unique=True)
    filename = Column(Text, nullable=False)
    docpath = Column(Text, nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="RESTRICT", onupdate="CASCADE"), nullable=False)
    metadata_json = Column("metadata", Text)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    pinned = Column(Boolean, server_default=text("false"))
    watched = Column(Boolean, server_default=text("false"))

    # 관계 정의
    vectors = relationship("DocumentVector", back_populates="document", cascade="all, delete-orphan")
    workspace = relationship("Workspace", back_populates="documents")

class DocumentVector(Base):
    __tablename__ = "document_vectors"
    __table_args__ = (UniqueConstraint("doc_id", "vector_id", name="document_vectors_doc_id_vector_id_key"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(Text, ForeignKey("workspace_documents.doc_id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    vector_id = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # 관계 정의
    document = relationship("WorkspaceDocument", back_populates="vectors")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    role = Column(Text, nullable=False, server_default=text("'user'"))
    username = Column(Text, nullable=False, unique=True)
    name = Column(Text, nullable=False)
    password = Column(Text, nullable=False)
    department = Column(Text, nullable=False)
    position = Column(Text, nullable=False)
    pfp_filename = Column(Text)
    bio = Column(Text, server_default=text("''"))
    daily_message_limit = Column(Integer)
    suspended = Column(Integer, nullable=False, server_default=text("0"))
    security_level = Column(Integer, nullable=False, server_default=text("3"))
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    expires_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # 관계 정의
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    workspace_chats = relationship("WorkspaceChat", back_populates="user")
    workspace_users = relationship("WorkspaceUser", back_populates="user")

class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id = Column(Text, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    expires_at = Column(DateTime, nullable=False)

    # 관계 정의
    user = relationship("User", back_populates="sessions")

class Workspace(Base):
    __tablename__ = "workspaces"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    slug = Column(Text, nullable=False, unique=True)
    category = Column(Text, nullable=False)  # CHECK constraint는 DB 레벨에서 처리
    vectorTag = Column(Text)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    temperature = Column(Float)
    chat_history = Column(Integer, nullable=False, server_default=text("20"))
    system_prompt = Column(Text)
    similarity_threshold = Column(Float, server_default=text("0.25"))
    provider = Column(Text)
    chat_model = Column(Text)
    top_n = Column(Integer, server_default=text("4"))
    chat_mode = Column(Text, nullable=False)  # CHECK constraint는 DB 레벨에서 처리
    pfp_filename = Column(Text)
    query_refusal_response = Column(Text)
    vector_search_mode = Column(Text, nullable=False, server_default=text("'hybrid'"))

    # 관계 정의
    documents = relationship("WorkspaceDocument", back_populates="workspace")
    chats = relationship("WorkspaceChat", back_populates="workspace")
    threads = relationship("WorkspaceThread", back_populates="workspace")
    workspace_users = relationship("WorkspaceUser", back_populates="workspace")

class WorkspaceChat(Base):
    __tablename__ = "workspace_chats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    thread_id = Column(Integer, ForeignKey("workspace_threads.id", ondelete="CASCADE", onupdate="CASCADE"))
    category = Column(Text, nullable=False)  # CHECK constraint는 DB 레벨에서 처리
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    include = Column(Boolean, nullable=False, server_default=text("true"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"))
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    feedback = Column(Integer)

    # 관계 정의
    user = relationship("User", back_populates="workspace_chats")
    workspace = relationship("Workspace", back_populates="chats")
    thread = relationship("WorkspaceThread", back_populates="chats")

class WorkspaceThread(Base):
    __tablename__ = "workspace_threads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    slug = Column(Text, nullable=False, unique=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    user_id = Column(Integer)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # 관계 정의
    workspace = relationship("Workspace", back_populates="threads")
    chats = relationship("WorkspaceChat", back_populates="thread")

class WorkspaceUser(Base):
    __tablename__ = "workspace_users"
    __table_args__ = (UniqueConstraint("user_id", "workspace_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

    # 관계 정의
    user = relationship("User", back_populates="workspace_users")
    workspace = relationship("Workspace", back_populates="workspace_users")

class LlmModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(Text, nullable=False)
    name = Column(Text, nullable=False, unique=True)
    revision = Column(Integer)
    model_path = Column(Text)
    category = Column(Text, nullable=False)  # CHECK constraint는 DB 레벨에서 처리
    subcategory = Column(Text)
    type = Column(Text, nullable=False, server_default=text("'base'"))  # CHECK constraint는 DB 레벨에서 처리
    is_default = Column(Boolean, nullable=False, server_default=text("false"))
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    trained_at = Column(DateTime)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class EmbeddingModel(Base):
    __tablename__ = "embedding_models"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False, unique=True)
    provider = Column(Text)
    model_path = Column(Text)
    is_active = Column(Integer, nullable=False, server_default=text("0"))
    activated_at = Column(DateTime)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False) 

# 추가 테이블들 (schema.sql의 고급 기능 포함)

class RagSettings(Base):
    """싱글톤 테이블 - RAG 전역 설정"""
    __tablename__ = "rag_settings"
    
    id = Column(Integer, primary_key=True, default=1)  # CHECK (id = 1)는 앱 레벨에서 처리
    search_type = Column(Text, nullable=False, server_default=text("'hybrid'"))
    chunk_size = Column(Integer, nullable=False, server_default=text("512"))
    overlap = Column(Integer, nullable=False, server_default=text("64"))
    embedding_key = Column(Text, nullable=False, server_default=text("'embedding_bge_m3'"))
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class SecurityLevelConfigTask(Base):
    """보안 레벨 설정 (작업 유형별)"""
    __tablename__ = "security_level_config_task"
    
    task_type = Column(Text, primary_key=True)  # 'doc_gen' | 'summary' | 'qna'
    max_level = Column(Integer, nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class SecurityLevelKeywordsTask(Base):
    """보안 레벨별 키워드 필터링"""
    __tablename__ = "security_level_keywords_task"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    task_type = Column(Text, nullable=False)
    level = Column(Integer, nullable=False)
    keyword = Column(Text, nullable=False)

# 누락된 테이블들 추가
class SystemPromptVariable(Base):
    __tablename__ = "system_prompt_variables"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    type = Column(Text, nullable=False)  # 'integer', 'text', 'datetime', 'float', 'bool'
    key = Column(Text, nullable=False)
    value = Column(Text)
    description = Column(Text, nullable=False)

class SystemPromptTemplate(Base):
    __tablename__ = "system_prompt_template"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False) # Business_Trip_Plan, Meeting_Minutes, Business_Trip_Report
    category = Column(Text, nullable=False)  # 'qa', 'doc_gen', 'summary'
    content = Column(Text, nullable=False) # 시스템 프롬프트 
    sub_content = Column(Text) # 유저 프롬프트
    required_vars = Column(Text)  # JSON 배열
    is_default = Column(Boolean, server_default=text("false"))
    is_active = Column(Boolean, server_default=text("true"))

class PromptMapping(Base):
    __tablename__ = "prompt_mapping"
    __table_args__ = (UniqueConstraint("template_id", "variable_id"),)
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("system_prompt_template.id", ondelete="CASCADE"), nullable=False)
    variable_id = Column(Integer, ForeignKey("system_prompt_variables.id", ondelete="CASCADE"), nullable=False) # 변하는 프롬프트트
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False) 

# 누락된 테이블들 추가
class CacheData(Base):
    __tablename__ = "cache_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    data = Column(Text, nullable=False)
    belongs_to = Column(Text)
    by_id = Column(Integer)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class EventLog(Base):
    __tablename__ = "event_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    event = Column(Text, nullable=False)
    metadata_json = Column("metadata", Text)
    user_id = Column(Integer)
    occurred_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class DocumentSyncQueue(Base):
    __tablename__ = "document_sync_queues"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_doc_id = Column(Integer, ForeignKey("workspace_documents.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False, unique=True)
    stale_after_ms = Column(Integer, nullable=False, server_default=text("604800000"))
    next_synced_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    last_synced_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class DocumentSyncExecution(Base):
    __tablename__ = "document_sync_executions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    queue_id = Column(Integer, ForeignKey("document_sync_queues.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    status = Column(Text, nullable=False, server_default=text("'unknown'"))
    result = Column(Text)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class PromptHistory(Base):
    __tablename__ = "prompt_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(Integer, ForeignKey("workspaces.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    prompt = Column(Text, nullable=False)
    modified_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL", onupdate="CASCADE"))
    modified_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class LlmTaskDefault(Base):
    __tablename__ = "llm_task_defaults"
    __table_args__ = (UniqueConstraint("category", "subcategory", name="ux_llm_task_defaults_cat_subcat_norm"),)
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(Text, nullable=False)  # 'qa', 'doc_gen', 'summary'
    subcategory = Column(Text)
    model_id = Column(Integer, ForeignKey("llm_models.id", ondelete="CASCADE", onupdate="CASCADE"), nullable=False)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class ChatFeedback(Base):
    __tablename__ = "chat_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(Text, nullable=False)  # 'qa', 'doc_gen', 'summary'
    chat_id = Column(Integer, ForeignKey("workspace_chats.id", ondelete="SET NULL", onupdate="CASCADE"))
    model_id = Column(Integer, ForeignKey("llm_models.id", ondelete="SET NULL", onupdate="CASCADE"))
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL", onupdate="CASCADE"))
    value = Column(Integer, nullable=False)  # 1 또는 -1
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class FineTuneDataset(Base):
    __tablename__ = "fine_tune_datasets"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    category = Column(Text, nullable=False)  # 'qa', 'doc_gen', 'summary'
    path = Column(Text, nullable=False)
    record_count = Column(Integer)
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    updated_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)

class FineTuneJob(Base):
    __tablename__ = "fine_tune_jobs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_job_id = Column(Text)
    dataset_id = Column(Integer, ForeignKey("fine_tune_datasets.id", ondelete="CASCADE", onupdate="CASCADE"))
    epochs = Column(Integer)
    learning_rate = Column(Float)
    batch_size = Column(Integer)
    prevent_overfit = Column(Boolean, server_default=text("false"))
    status = Column(Text, nullable=False)  # 'queued', 'running', 'succeeded', 'failed'
    metrics = Column(Text)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)

class FineTunedModel(Base):
    __tablename__ = "fine_tuned_models"
    __table_args__ = (
        Index("ix_fine_tuned_models_base_model_id", "base_model_id"),
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey("llm_models.id", ondelete="CASCADE", onupdate="CASCADE"))
    job_id = Column(Integer, ForeignKey("fine_tune_jobs.id", ondelete="CASCADE", onupdate="CASCADE"))
    provider_model_id = Column(Text, nullable=False)
    lora_weights_path = Column(Text)
    type = Column(Text, nullable=False)  # 'base', 'lora', 'full'
    is_active = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    
    # 베이스 모델 참조 (LoRA/QLoRA용)
    base_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=True)
    base_model_path = Column(Text, nullable=True)  # 상대경로(백엔드 루트 기준) 