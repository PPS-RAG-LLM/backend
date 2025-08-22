```
// RAG LLM Backend Database ERD
Table users {
  id                  integer   [primary key, not null, increment]
  username            text      [not null] // ID
  name                text      [not null] // 진짜 이름
  password            text      [not null]
  role                text      [not null, default: 'default']
  deportment          text      [not null] // 부서
  position            text      [not null]  // 직책
  pfp_filename        text                                      // 프로필 사진 이름
  seen_recovery_codes boolean   [default: false]
  daily_message_limit integer
  bio                 text      [default: '']                   // biography
  security_number     integer   [not null, default: 1, note: "CHECK (1,2,3)"]   // 보안 등급
  created_at          datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at          datetime  [not null, default: `CURRENT_TIMESTAMP`] 
  expires_at          datetime  [not null, default: `CURRENT_TIMESTAMP`]        // 퇴사일
  suspended           integer   [not null, default: 0]      // 0 정상 / 1 정지 또는 잠금
}


// QA, 문서요약, 문서생성 
Table workspaces {
  id                      integer [primary key, not null, increment]
  name                    text    [not null]
  slug                    text    [not null]
  category                text    [not null, note: "CHECK (category IN ('qa','doc_gen','summary'))"]
  vectorTag               text 

  created_at              datetime [not null, default: `CURRENT_TIMESTAMP`]
  updated_at              datetime [not null, default: `CURRENT_TIMESTAMP`]
  
  temp                    float   // real 소수점
  chat_history            integer [not null, default: 20]
  similarity_threshold    float   [default: 0.25]
  
  chat_model              text 
  top_n                   integer [default : 4, note:"CHECK topN > 0"]
  chat_mode               text    [default: 'chat', note: "CHECK (chat_mode IN('chat', 'query'))"] // chat | query

  pfp_filename            text    // 프로필 사진 이름
  provider                text    //openai, huggingface, etc,.

  query_refusal_response  text
  vector_search_mode      text    [default: 'default']
}


Table workspace_users {
  id              integer   [primary key, not null, increment]
  user_id         integer   [not null]
  workspace_id    integer   [not null]
  created_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref : workspace_users.user_id       > users.id      [delete: cascade, update: cascade]
Ref : workspace_users.workspace_id  > workspaces.id [delete: cascade, update: cascade]


// category 중 qa 만 threads를 가짐.
Table workspace_threads {
  id              integer   [not null, primary key, increment]
  name            text      [not null]
  slug            text      [not null]
  workspace_id    integer   [not null]
  user_id         integer
  created_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref : workspace_threads.workspace_id > workspaces.id  [delete: cascade, update: cascade]
Ref : workspace_threads.user_id      > users.id       [delete:cascade, update: cascade]


Table workspace_documents {
  id            integer   [primary key, not null, increment]
  doc_id        text      [not null]
  filename      text      [not null]
  docpath       text      [not null]
  workspace_id  integer   [not null]
  metadata      text
  created_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
  pinned        bool      [default: false]
  watched       bool      [default: false]
}
Ref : workspace_documents.workspace_id > workspaces.id [delete : cascade, update: cascade]


Table workspace_chats {
  id              integer   [primary key, not null, increment]
  workspace_id    integer   [not null]
  category        text      [not null, note: "CHECK (value IN ('qa','doc_gen','summary'))"]
  prompt          text      [not null]
  response        text      [not null]
  include         bool      [not null, default: true]
  user_id         integer
  created_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at      datetime  [not null, default: `CURRENT_TIMESTAMP`]
  thread_id       integer   // QA만 쓰레드를 가질 수 있음.
  feedback        integer   [note: "CHECK (feedback IN (1,-1))"]
}
Ref : workspace_chats.user_id       > users.id              [delete: cascade, update: cascade]
//Add FOREIGN KEY 
Ref : workspace_chats.workspace_id  > workspaces.id         [delete: cascade, update: cascade]
Ref : workspace_chats.thread_id     > workspace_threads.id  [delete: cascade, update: cascade]


Table workspace_prompt_history {
  id            integer   [pk, not null, increment]
  workspace_id  integer   [not null]
  prompt        text      [not null]
  modified_by   integer
  modified_at   datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref : workspace_prompt_history.workspace_id > workspaces.id [delete: cascade, update: cascade]
Ref : workspace_prompt_history.modified_by  > users.id      [delete: cascade, update: cascade]

Table workspace_prompt_templates {
  id            integer   [pk, not null, increment]
  workspace_id  integer   [not null, ref : > workspaces.id]
  template_id   integer   [ not null, ref: > system_prompt_template.id]
  is_default    bool      [default: false]
  created_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
  category     text     [not null, note: "CHECK (category IN ('qa','doc_gen','summary'))"]
  Indexes {
    (workspace_id, category, is_default) [unique, name: 'idx_ws_cat_default']
  }
}

///////////////////// 문서 임베딩 관련 /////////////////////
// workspace_documents 테이블 포함
// 어떤 문서를 어제 다시 임베딩(동기화)해야하는지를 스케줄링하는 할일 목록
Table document_sync_queues {
  id                integer   [pk, not null, increment]
  workspace_doc_id  integer   [not null]                              // 대상문서 - 한건당 한행 존재
  stale_after_ms    integer   [not null, default: 604800000]          //  문서가 얼마나 오래되면 ‘낡았다(stale)’고 간주할지(기본 604 800 000 ms = 7일).
  next_synced_at    datetime  [not null]
  created_at        datetime  [not null, default: `CURRENT_TIMESTAMP`]
  last_synced_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref: document_sync_queues.workspace_doc_id > workspace_documents.id [delete: cascade, update: cascade]


// 문서-동기화 작업이 실제로 언제·어떤 결과로 실행됐는지 남기는 실행-이력 테이블
Table document_sync_executions {
  id            integer   [pk, not null, increment]
  queue_id      integer   [not null]
  status        text      [not null, default: 'unknown']
  result        text
  created_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref : document_sync_executions.queue_id > document_sync_queues.id [delete: cascade, update: cascade]


Table document_vectors {
  id            integer   [pk, not null, increment]
  doc_id        text      [not null]
  vector_id     text      [not null]
  created_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
}

////////////////////////////////////////////////////////////////////////////////////////////////////


// 감사(Audit)·이력·모니터링용 로그 테이블
// 예 : event : workspace_created / metadata : {"workspaceName": "pps"}
Table event_logs {
  id          integer   [pk, not null, increment]
  event       text      [not null]                // 이벤트 이름 or 종류  (login,file_updload, prompt_edit)
  metadata    text                                // 이벤트 세부 정보     (보통은 json문자열 직렬화 해서 넣음 )
  user_id     integer                             // 이벤트를 발생시킨 사용자가 없으면 null
  occurred_at datetime  [not null, default: `CURRENT_TIMESTAMP`]
}

// 임시(캐시) 데이터를 DB에 보관
// 주기적으로 DELETE FROM cache_data WHERE expiresAt IS NOT NULL AND expiresAt < NOW(); 같은 정리 작업을 돌려야 DB가 불어나지 않음
Table cache_data {
  id          integer   [pk, not null, increment]
  name        text      [not null]
  data        text      [not null]
  belongs_to  text      // 이 캐시가 어떤 영역(테이블·기능·워크스페이스 등)에 속하는지 태그로 표시.
  by_id       integer   // belongsTo 가 가리키는 table의 ID.
  expires_at  datetime  
  created_at  datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at  datetime  [not null, default: `CURRENT_TIMESTAMP`]
}

///////////////////// 문서 생성 및 요약 관련 테이블 /////////////////////
// 프롬프트 템플릿에 삽입되는 “치환 변수(placeholder)”와 그 기본값/설명을 저장합니다.
Table system_prompt_variables {
  id          integer   [pk, not null, increment]
  type        text      [not null, note: "CHECK (type IN ('integer', 'text', 'datetime', 'float', 'bool'))"]
  key         text      [not null]      // 변수명(예: '{{user_name}}', '{{today}}')
  value       text                      // 기본 치환값(예: ‘홍길동’, ‘2025-07-29’) 또는 JSON 문자열
  description text      [not null]      // 변수의 의미, 
}

// 프롬프트 탬플릿
Table system_prompt_template {
  id            integer [pk, not null, increment]
  name          text    [not null]          // 화면표시용: ‘출장계획서’
  category      text    [not null, note : "CHECK (category IN ('doc_gen', 'summary', 'qa'))"] 
  content       text    [not null]          // 실제 프롬프트 본문
  required_vars text                        // JSON 배열: ["date","name"] 
  is_active     bool    [default: true]
}

// 템플릿 변수 N:M 매핑
Table prompt_mapping {
  id          integer   [pk, not null, increment]
  template_id integer 
  variable_id integer 
  created_at  datetime  [not null, default: `CURRENT_TIMESTAMP`]
  updated_at  datetime  [not null, default: `CURRENT_TIMESTAMP`]
}
Ref : prompt_mapping.template_id > system_prompt_template.id  [delete: cascade, update: cascade]
Ref : prompt_mapping.variable_id > system_prompt_variables.id [delete: cascade, update: cascade]



///////////////////// 파인튜닝 관련 테이블 /////////////////////
// 모델 관리 테이블
Table llm_models {
  id            integer [pk, increment]            // 내부 PK
  provider      text    [not null]                 // openai | hf | vertex …
  name          text    [unique, not null]         // 사람이 읽는 이름
  revision      integer [default: 0]               // 파인튜닝 버전 (0이면 파인튜닝 안된 오리지널 버전)
  model_path    text                               // 모델 파일/가중치 위치 (경로 또는 url)
  category      text    [not null, note: "CHECK (category IN ('qa', 'doc_gen', 'summary'))"]
  type          text    [not null, default: 'base', note: "CHECK (type IN ('base', 'lora', 'full'))"]
  is_active     boolean [default: true]
  trained_at    datetime
  created_at    datetime [default: `CURRENT_TIMESTAMP`]
}


// 사용자 피드백 수집
Table chat_feedback {
  id            integer   [pk, not null,increment]
  chat_id       integer   [ref: > workspace_chats.id]   
  model_id      integer   [ref: > llm_models.id]
  user_id       integer   [ref: > users.id]
  value         integer   [not null, note: "CHECK (value IN (1,-1))"] // 1=like, −1=dislike
  created_at    datetime  [not null, default: `CURRENT_TIMESTAMP`]
}


Table fine_tune_datasets {
  id           integer [pk, not null, increment]
  name         text    [not null]           // 데이터셋 이름
  description  text                         // 설명
  path         text    [not null]           // 저장 위치
  record_count integer                      // 레코드(샘플) 개수
  created_at   datetime [not null, default: `CURRENT_TIMESTAMP`]
  updated_at   datetime [not null, default: `CURRENT_TIMESTAMP`]
}


// 1. 파인튜닝 실행(Job) – 비동기 작업 추적 */
Table fine_tune_jobs {
  id              integer   [pk, not null, increment]                
  provider_job_id text                                      // OpenAI, HF 등 외부 Job ID
  dataset_id      integer   [ref: > fine_tune_datasets.id]  // 학습 데이터
  hyperparameters text                                      // JSON: epoch, batch_size, learning_rate...
  status          text      [not null, note: "CHECK (status IN ('queued', 'running', 'succeeded', 'failed'))"] 
  metrics         text                                      // JSON: val_loss, accuracy …
  started_at      datetime
  finished_at     datetime
 }


/* 2. 파인튜닝 결과(Model) – llm_models에 INSERT + 추가 메타 데이터 */
Table fine_tuned_models {
  id                integer   [pk, not null, increment]
  model_id          integer   [ref: > llm_models.id]        // llm_models 에 등록된 최종 모델
  job_id            integer   [ref: > fine_tune_jobs.id]    // 어떤 Job의 산출물인가
  provider_model_id text      [not null]                    // 외부 모델 ID
  lora_weights_path text                                    // LoRA 파일 경로(옵션)
  type              text      [not null, note: "CHECK (value IN('base', 'lora', 'full'))"] 
  is_active         boolean   [not null, default: true]
  created_at        datetime  [not null, default:`CURRENT_TIMESTAMP`]
}



```