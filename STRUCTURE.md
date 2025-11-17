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
- 모델 파일(Qwen2.5-7B-Instruct-1M)은 models/qwen/loader.py에서 로드해서 사용.
- API 서버는 src/api/main.py에서 실행.
- 각 기능별 라우트(routes/)와 비즈니스 로직(core/)을 분리해서 관리.
- 관리자 페이지는 별도의 프론트엔드(React, Vue 등)에서 API 호출로 구현(백엔드는 routes/admin.py 등에서 처리).
- 문서/요약/템플릿 등은 각각의 라우트와 로직으로 분리.


# 3. 확장성/유지보수성 포인트
모델 교체/추가가 쉬움 (models/qwen/loader.py만 수정)
기능별 코드 분리로 협업/유지보수 용이
테스트 코드(tests/)로 안정성 확보
환경설정 분리(config.py)




