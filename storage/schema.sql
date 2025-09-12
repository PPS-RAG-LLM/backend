CREATE TABLE IF NOT EXISTS "users" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "role" TEXT NOT NULL DEFAULT 'user',
    "username" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "password" TEXT NOT NULL,
    "department" TEXT NOT NULL,
    "position" TEXT NOT NULL,
    "pfp_filename" TEXT,
    "bio" TEXT DEFAULT '',
    "daily_message_limit" INTEGER,
    "suspended" INTEGER NOT NULL DEFAULT 0 CHECK ("suspended" IN (0, 1)),
    "security_level" INTEGER NOT NULL DEFAULT 3 CHECK ("security_level" IN (1, 2, 3)),
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS "users_username_key" ON "users" ("username");

CREATE TABLE IF NOT EXISTS "user_sessions" (
    "session_id" TEXT PRIMARY KEY,
    "user_id" INTEGER NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "expires_at" DATETIME NOT NULL,
    CONSTRAINT "user_sessions_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "workspace_documents" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "doc_id" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "docpath" TEXT NOT NULL,
    "workspace_id" INTEGER NOT NULL,
    "metadata" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "pinned" BOOLEAN DEFAULT false,
    "watched" BOOLEAN DEFAULT false,
    CONSTRAINT "workspace_documents_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_documents_doc_id_key" ON "workspace_documents" ("doc_id");
--

CREATE TABLE IF NOT EXISTS "document_vectors" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "doc_id" TEXT NOT NULL,
    "vector_id" TEXT NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "document_vectors_doc_id_fkey" FOREIGN KEY ("doc_id") REFERENCES "workspace_documents" ("doc_id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "document_vectors_doc_id_vector_id_key" ON "document_vectors" ("doc_id", "vector_id");

CREATE TABLE IF NOT EXISTS "workspaces" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "category" TEXT NOT NULL CHECK (
        "category" IN ('qa', 'doc_gen', 'summary')
    ),
    "vectorTag" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "temperature" float,
    "chat_history" INTEGER NOT NULL DEFAULT 20,
    "system_prompt" TEXT,
    "similarity_threshold" float DEFAULT 0.25,
    "provider" TEXT,
    "chat_model" TEXT,
    "top_n" INTEGER DEFAULT 4 CHECK ("top_n" > 0),
    "chat_mode" TEXT NOT NULL CHECK (
        "chat_mode" IN ('chat', 'query')
    ),
    "pfp_filename" TEXT,
    "query_refusal_response" TEXT,
    "vector_search_mode" TEXT NOT NULL DEFAULT 'hybrid' CHECK (
        "vector_search_mode" IN ('hybrid', 'semantic', 'bm25')
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspaces_slug_key" ON "workspaces" ("slug");

CREATE TABLE IF NOT EXISTS "workspace_chats" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "workspace_id" INTEGER NOT NULL,
    "thread_id" INTEGER,
    "category" TEXT NOT NULL CHECK (
        "category" IN ('qa', 'doc_gen', 'summary')
    ),
    "prompt" TEXT NOT NULL,
    "response" TEXT NOT NULL,
    "include" bool NOT NULL DEFAULT true,
    "user_id" INTEGER,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "feedback" INTEGER,
    CONSTRAINT "workspace_chats_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "workspace_chats_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "workspace_chats_thread_id_fkey" FOREIGN KEY ("thread_id") REFERENCES "workspace_threads" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "cache_data" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL,
    "data" TEXT NOT NULL,
    "belongs_to" TEXT,
    "by_id" INTEGER,
    "expires_at" DATETIME,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS "event_logs" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "event" TEXT NOT NULL,
    "metadata" TEXT,
    "user_id" INTEGER,
    "occurred_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS "event_logs_event_idx" ON "event_logs" ("event");

CREATE TABLE IF NOT EXISTS "workspace_threads" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL,
    "slug" TEXT NOT NULL,
    "workspace_id" INTEGER NOT NULL,
    "user_id" INTEGER,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "workspace_threads_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "workspace_threads_slug_key" ON "workspace_threads" ("slug");

CREATE INDEX IF NOT EXISTS "workspace_threads_workspace_id_idx" ON "workspace_threads" ("workspace_id");

CREATE INDEX IF NOT EXISTS "workspace_threads_user_id_idx" ON "workspace_threads" ("user_id");

CREATE TABLE IF NOT EXISTS "document_sync_queues" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "workspace_doc_id" INTEGER NOT NULL,
    "stale_after_ms" INTEGER NOT NULL DEFAULT 604800000,
    "next_synced_at" DATETIME NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "last_synced_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "document_sync_queues_workspace_doc_id_fkey" FOREIGN KEY ("workspace_doc_id") REFERENCES "workspace_documents" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS "document_sync_queues_workspace_doc_id_key" ON "document_sync_queues" ("workspace_doc_id");

CREATE INDEX IF NOT EXISTS "document_sync_queues_next_synced_at_idx" ON "document_sync_queues" ("next_synced_at");

CREATE TABLE IF NOT EXISTS "document_sync_executions" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "queue_id" INTEGER NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'unknown',
    "result" TEXT,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "document_sync_executions_queue_id_fkey" FOREIGN KEY ("queue_id") REFERENCES "document_sync_queues" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "workspace_users" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "user_id" INTEGER NOT NULL,
    "workspace_id" INTEGER NOT NULL,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE ("user_id", "workspace_id"),
    CONSTRAINT "workspace_users_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "workspace_users_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "prompt_history" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "workspace_id" INTEGER NOT NULL,
    "prompt" TEXT NOT NULL,
    "modified_by" INTEGER,
    "modified_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "prompt_history_workspace_id_fkey" FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "prompt_history_modified_by_fkey" FOREIGN KEY ("modified_by") REFERENCES "users" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS "prompt_history_workspace_id_idx" ON "prompt_history" ("workspace_id");

CREATE TABLE IF NOT EXISTS "system_prompt_variables" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "type" TEXT NOT NULL CHECK (
        "type" IN (
            'integer',
            'text',
            'datetime',
            'float',
            'bool'
        )
    ),
    "key" TEXT NOT NULL,
    "value" TEXT,
    "description" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS "system_prompt_template" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL, -- 화면표시용: ‘출장계획서’
    "category" TEXT NOT NULL CHECK (
        category IN ('qa', 'doc_gen', 'summary')
    ),
    "content" TEXT NOT NULL, -- 실제 프롬프트 본문
    "required_vars" TEXT, -- JSON 배열: ["date","name"] 
    "is_default" BOOLEAN DEFAULT false,
    "is_active" BOOLEAN DEFAULT true
);

-- system_prompt_template: 카테고리별 기본 템플릿은 하나만 허용
CREATE UNIQUE INDEX IF NOT EXISTS "system_prompt_template_one_default_per_category" ON "system_prompt_template" ("category")
WHERE
    is_default = true;

-- 기본값 자동 전환: 새 기본값 삽입 시 동일 카테고리 기존 기본값 해제
CREATE TRIGGER IF NOT EXISTS trg_spt_before_insert_single_default
BEFORE INSERT ON system_prompt_template
FOR EACH ROW
WHEN NEW.is_default = 1
BEGIN
  UPDATE system_prompt_template
  SET is_default = 0
  WHERE category = NEW.category;
END;

-- 기본값 자동 전환: 기본값으로 업데이트될 때 동일 카테고리 기존 기본값 해제
CREATE TRIGGER IF NOT EXISTS trg_spt_before_update_single_default
BEFORE UPDATE OF is_default, category ON system_prompt_template
FOR EACH ROW
WHEN NEW.is_default = 1 AND (OLD.is_default IS NOT 1 OR NEW.category <> OLD.category)
BEGIN
  UPDATE system_prompt_template
  SET is_default = 0
  WHERE category = NEW.category AND id <> OLD.id;
END;

CREATE TABLE IF NOT EXISTS "prompt_mapping" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "template_id" INTEGER NOT NULL,  -- NOT NULL로 변경 (필수 관계) [1:N]
  "variable_id" INTEGER NOT NULL,  -- NOT NULL로 변경 (필수 관계) [N:1]
  "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  -- 외래키 제약조건 추가
  CONSTRAINT "prompt_mapping_template_id_fkey" 
    FOREIGN KEY ("template_id") 
    REFERENCES "system_prompt_template" ("id") 
    ON DELETE CASCADE ON UPDATE CASCADE,  -- 템플릿 삭제 시 매핑도 삭제
  CONSTRAINT "prompt_mapping_variable_id_fkey" 
    FOREIGN KEY ("variable_id") 
    REFERENCES "system_prompt_variables" ("id") 
    ON DELETE CASCADE ON UPDATE CASCADE,  -- 변수 삭제 시 매핑도 삭제
  -- 동일한 템플릿-변수 조합 중복 방지
  UNIQUE("template_id", "variable_id")
);

CREATE TABLE IF NOT EXISTS "llm_models" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "provider" TEXT NOT NULL,
  "name" TEXT UNIQUE NOT NULL,
  "revision" INTEGER,
  "model_path" TEXT,
  "category" TEXT NOT NULL CHECK (category IN ('qa', 'doc_gen', 'summary', 'all')),
  "subcategory" TEXT, -- doc_gen 전용(옵션)
  "type" TEXT NOT NULL DEFAULT 'base' CHECK ("type" IN ('base', 'lora', 'full')),
  "is_default" BOOLEAN NOT NULL DEFAULT false,
  "is_active"  BOOLEAN NOT NULL DEFAULT true,
  "trained_at" DATETIME,
  "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
-- 카테고리별 is_default=1 모델은 하나만 허용
CREATE UNIQUE INDEX IF NOT EXISTS "llm_models_one_default_per_category" ON "llm_models" ("category")
WHERE
    is_default = true;

-- 기본값 자동 전환: 새 기본 모델 삽입 시 동일 카테고리 기존 기본값 해제
CREATE TRIGGER IF NOT EXISTS trg_llm_before_insert_single_default
BEFORE INSERT ON llm_models
FOR EACH ROW
WHEN NEW.is_default = 1
BEGIN
  UPDATE llm_models
  SET is_default = 0
  WHERE category = NEW.category;
END;

-- 기본값 자동 전환: 기본값으로 업데이트될 때 동일 카테고리 기존 기본값 해제
CREATE TRIGGER IF NOT EXISTS trg_llm_before_update_single_default
BEFORE UPDATE OF is_default, category ON llm_models
FOR EACH ROW
WHEN NEW.is_default = 1 AND (OLD.is_default IS NOT 1 OR NEW.category <> OLD.category)
BEGIN
  UPDATE llm_models
  SET is_default = 0
  WHERE category = NEW.category AND id <> OLD.id;
END;

-- (신규) 테스크 기본 모델 매핑: (category, subcategory) 별 1개 모델을 지정
CREATE TABLE IF NOT EXISTS "llm_task_defaults" (
  "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
  "category" TEXT NOT NULL CHECK (category IN ('qa', 'doc_gen', 'summary')),
  "subcategory" TEXT,
  "model_id" INTEGER NOT NULL,
  "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT "llm_task_defaults_model_id_fkey"
    FOREIGN KEY ("model_id") REFERENCES "llm_models" ("id")
    ON DELETE CASCADE ON UPDATE CASCADE
);

-- (표현식 허용) 카테고리+서브카테고리(NULL을 빈문자 취급)의 고유성 보장
CREATE UNIQUE INDEX IF NOT EXISTS "ux_llm_task_defaults_cat_subcat_norm"
ON "llm_task_defaults" (
  "category",
  IFNULL("subcategory",'')
);

CREATE TABLE IF NOT EXISTS "chat_feedback" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "category" TEXT NOT NULL CHECK (
        "category" IN ('qa', 'doc_gen', 'summary')
    ),
    "chat_id" INTEGER, -- NULL 허용
    "model_id" INTEGER, -- NULL 허용
    "user_id" INTEGER, -- NULL 허용
    "value" INTEGER NOT NULL CHECK ("value" IN (1, -1)),
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "chat_feedback_chat_id_fkey" FOREIGN KEY ("chat_id") REFERENCES "workspace_chats" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "chat_feedback_model_id_fkey" FOREIGN KEY ("model_id") REFERENCES "llm_models" ("id") ON DELETE SET NULL ON UPDATE CASCADE,
    CONSTRAINT "chat_feedback_user_id_fkey" FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE SET NULL ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "fine_tune_datasets" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "name" TEXT NOT NULL, -- 데이터셋 파일 이름
    "category" TEXT NOT NULL CHECK (
        "category" IN ('qa', 'doc_gen', 'summary')
    ),
    "path" TEXT NOT NULL, -- 데이터셋 파일 경로
    "record_count" INTEGER, -- 데이터셋 파일 행 수 
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 비동기 튜닝 작업 관리
CREATE TABLE IF NOT EXISTS "fine_tune_jobs" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "provider_job_id" TEXT,
    "dataset_id" INTEGER,
    "epochs" INTEGER,
    "learning_rate" FLOAT,
    "batch_size" INTEGER,
    "prevent_overfit" BOOLEAN DEFAULT false,
    "status" TEXT NOT NULL CHECK (
        "status" IN (
            'queued',
            'running',
            'succeeded',
            'failed'
        )
    ),
    "metrics" TEXT, -- 튜닝 작업 결과 메트릭
    "started_at" DATETIME,
    "finished_at" DATETIME,
    CONSTRAINT "fine_tune_jobs_dataset_id_fkey" FOREIGN KEY ("dataset_id") REFERENCES "fine_tune_datasets" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS "fine_tuned_models" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "model_id" INTEGER, -- 기존 모델 ID
    "job_id" INTEGER, -- 튜닝 작업 ID 
    "provider_model_id" TEXT NOT NULL,
    "lora_weights_path" TEXT,
    "type" TEXT NOT NULL CHECK (
        "type" IN ('base', 'lora', 'full')
    ),
    "is_active" boolean NOT NULL DEFAULT true,
    "created_at" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT "fine_tuned_models_model_id_fkey" FOREIGN KEY ("model_id") REFERENCES "llm_models" ("id") ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT "fine_tuned_models_job_id_fkey" FOREIGN KEY ("job_id") REFERENCES "fine_tune_jobs" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

-- =========================
-- Embedding models (하나만 활성)
-- =========================
CREATE TABLE IF NOT EXISTS embedding_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE, -- 모델 키(예: bge, embedding_bge_m3, qwen3_0_6b)
    provider TEXT, -- 선택(예: hf, ollama, local 등)
    model_path TEXT, -- 로컬 경로나 식별자(옵션)
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
    activated_at DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- "활성 = 1" 이 최대 1행만 되도록 부분 유니크 인덱스
CREATE UNIQUE INDEX IF NOT EXISTS ux_embedding_models_active_one ON embedding_models (is_active)
WHERE
    is_active = 1;

-- -- 최소 한 개 기본 후보를 만들어두고 싶으면(선택):
-- INSERT OR IGNORE INTO embedding_models(name, provider, model_path, is_active, activated_at)
-- VALUES ('qwen3_0_6b', 'local', NULL, 1, CURRENT_TIMESTAMP);

-- =========================
-- Vector settings (검색/청크)
-- =========================
CREATE TABLE IF NOT EXISTS vector_settings (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    search_type TEXT NOT NULL DEFAULT 'hybrid' CHECK (
        search_type IN ('hybrid', 'semantic', 'bm25')
    ),
    chunk_size INTEGER NOT NULL DEFAULT 512,
    overlap INTEGER NOT NULL DEFAULT 64,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT
    OR IGNORE INTO vector_settings (
        id,
        search_type,
        chunk_size,
        overlap
    )
VALUES (1, 'hybrid', 512, 64);

-- =========================
-- Security level rules (작업유형별)
-- =========================
CREATE TABLE IF NOT EXISTS security_level_config_task (
    task_type TEXT PRIMARY KEY, -- 'doc_gen' | 'summary' | 'qna'
    max_level INTEGER NOT NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS security_level_keywords_task (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,
    level INTEGER NOT NULL,
    keyword TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_slk_task_level ON security_level_keywords_task(task_type, level, keyword);

-- =========================
-- RAG global settings (single source of truth)
-- =========================
CREATE TABLE IF NOT EXISTS rag_settings (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  search_type   TEXT NOT NULL DEFAULT 'hybrid' CHECK (search_type IN ('hybrid','bm25','vector')),
  chunk_size    INTEGER NOT NULL DEFAULT 512 CHECK (chunk_size > 0),
  overlap       INTEGER NOT NULL DEFAULT 64 CHECK (overlap >= 0),
  embedding_key TEXT NOT NULL DEFAULT 'embedding_bge_m3',
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO rag_settings(id) VALUES (1)
ON CONFLICT(id) DO NOTHING;
