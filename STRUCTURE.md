> Qwen2.5-7B-Instruct-1M 모델을 활용해서 챗봇, 관리자 페이지, 문서 관련 기능(요약, 템플릿 등)을 고려한 설계 방향

# 1. 디렉토리 구조 설계
```
backend/                              # Python(FastAPI) 백엔드 - REST/RAG 서비스
├── Dockerfile                        # 백엔드 전용 컨테이너 빌드
├── requirements.txt                  # (Poetry 미사용 시) 의존성 고정
├── config.py                         # Settings(BaseSettings) - .env 로딩
├── main.py                           # `uvicorn main:app` 진입점 래퍼
├── routers/                          # 기능별 api 라우터         
│   ├── users/                 
│   │   ├── doc_generation.py         # 문서 생성 
│   │   ├── doc_summary.py            # 문서 요약
│   │   ├── setting_workspace.py      # 워크스페이스 설정 업데이트 (Profile, QA, Vector Search)
│   │   └── qna.py                     # Q&A
│   └── admin/                 
│       ├── manage_prompts_api.py     # 프롬프트 관리
│       └── fine_tuning_api.py        # 모델 파인튜닝 관련
│
├── service/                          # 비즈니스 로직, 도메인 규칙 적용하는 계층
│   ├── users/                        # 사용자 비즈니스 로직
│   │   ├── doc_generation.py         # 다양한 파일 형식 파싱·저장 로직
│   │   ├── setting_workspace.py      # 워크스페이스 설정 업데이트 (Profile, QA, Vector Search)
│   │   └── qna.py                     # RAG QA 파이프라인
│   └── admin/                 
│       ├── manage_prompts.py         # 프롬프트 관리
│       ├── manage_model_ERP.py       # ERP 관리
│       └── fine_tuning.py            # 모델 파인튜닝 관련
│
├── repository/                    #  DB, 외부 저장소, API 호출 등 데이터 접근 계층 → CRUD 직접 처리
│   ├── users/                 
│   │   ├── qna.py             
│   │   └── …          
│   ├── admin/                 
│   │    ├── prompts.py         
│   │    └── … 
│   └── vector_repo.py             # Milvus 컬렉션 생성·삽입·검색·삭제를 담당 
│
├── utils/                     # 공통 헬퍼 
|   ├── logger.py              # 로깅
|   ├── timestamp.py           # 타임스탬프
|   ├── …   
|   └── llms/
|        ├── base.py                     # Model factory 
|        ├── huggingface/                # provider
│        |    ├── qwen/                  # HF : 각 모델 family
│        |    |     ├── qwen_vl_7b.py    # 실제 모델 이름  
|        |    |     └── qwen_7b.py
|        |    ├── google/
|        |    |     └── gemma3_12b.py
|        |    └── openai/     
|        |          └── oss_20b.py
|        ├── openai/
|        |    └── ***.py
|        └── ollama/
|             └── ***.py 
│
├── storage/                   # 데이터 베이스 
|   ├── pps_rag.db             # SQLite DB
|   └── db.sql                 # SQL 문
│
├── tests/                     
# 백엔드 테스트 폴더             
│   ├── test_routes.py
│   └── test_core.py
└── README.md
```

# 2. 서빙 구조
- 모델 파일(Gemma3-27B)은 models/qwen/loader.py에서 로드해서 사용.
- API 서버는 src/api/main.py에서 실행.
- 각 기능별 라우트(routes/)와 비즈니스 로직(core/)을 분리해서 관리.
- 관리자 페이지는 별도의 프론트엔드(React, Vue 등)에서 API 호출로 구현(백엔드는 routes/admin.py 등에서 처리).
- 문서/요약/템플릿 등은 각각의 라우트와 로직으로 분리.


# 3. 확장성/유지보수성 포인트
모델 교체/추가가 쉬움 (models/qwen/loader.py만 수정)
기능별 코드 분리로 협업/유지보수 용이
테스트 코드(tests/)로 안정성 확보
환경설정 분리(config.py)




### 2. 아키텍처 설명 (보고서/산출물용)
- 산출물 문서에 넣을 때는 아래 4가지 계층으로 설명하면 명확합니다.

1) Presentation Layer (Frontend)
    - 사용자가 접근하는 웹 애플리케이션입니다.
    - 백엔드 API 서버와 REST API로 통신합니다.
2) Application Layer (Backend - FastAPI)
    - API Router: 클라이언트의 요청을 받아 적절한 서비스로 라우팅합니다. (routers/)
    - Service & Logic:
    - RAG Core: 문서 파싱, 임베딩, 검색(Retrieval), 답변 생성(Generation)을 수행합니다.
    - Admin Features: 파인튜닝, 프롬프트 관리 등을 수행합니다.
    - LLM Loader: Qwen, OpenAI 등 다양한 모델을 로드하고 추론을 수행하는 모듈입니다.
3) Data Layer
    - PostgreSQL: 사용자 정보(User), 채팅 세션(Session), 문서 메타데이터 등을 저장하는 관계형 데이터베이스입니다.
    - Milvus: 문서 내용을 벡터화하여 저장하고, 유사도 및 하이브리드 검색을 수행하는 벡터 데이터베이스입니다.
    - File Storage: 업로드된 원본 문서(PDF, DOCX)와 파인튜닝된 모델 가중치(Weights)를 저장합니다.
4) External Interface   
    - SSO: 사내 통합 인증 시스템과 연동합니다.
    - External LLM: 필요 시 OpenAI 등 외부 API를 호출합니다.