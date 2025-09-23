BEGIN;

INSERT INTO
    "llm_models" (
        "id",
        "provider",
        "name",
        "revision",
        "model_path",
        "category",
        "type",
        "is_default",
        "is_active",
        "trained_at",
        "created_at"
    )
VALUES (
        1,
        'huggingface',
        'gpt-oss-20b',
        0,
        './service/storage/model/gpt-oss-20b',
        'all',
        'base',
        1,
        1,
        '2025-09-03 07:52:05',
        '2025-09-03 07:52:05'
    );

INSERT INTO
    "llm_models" (
        "id",
        "provider",
        "name",
        "revision",
        "model_path",
        "category",
        "type",
        "is_default",
        "is_active",
        "trained_at",
        "created_at"
    )
VALUES (
        2,
        'huggingface',
        'Qwen2.5-7B-Instruct-1M',
        0,
        './service/storage/model/Qwen2.5-7B-Instruct-1M',
        'all',
        'base',
        1,
        1,
        '2025-09-03 07:52:05',
        '2025-09-03 07:52:05'
    );

INSERT INTO
    "rag_settings" (
        "id",
        "search_type",
        "chunk_size",
        "overlap",
        "embedding_key",
        "updated_at"
    )
VALUES (
        1,
        'hybrid',
        512,
        64,
        'embedding_qwen3_0_6b',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "system_prompt_template" (
        "id",
        "name",
        "category",
        "content",
        "sub_content",
        "required_vars",
        "is_default",
        "is_active"
    )
VALUES (
        5,
        '기본 요약 프롬프트',
        'summary',
        '당신은 요약 전문 AI 어시스턴트 입니다. 사용자가 요청한 [Context]를 [USER_PROMPT]에 따라 요약하세요.',
        '',
        '[]',
        1,
        1
    );

INSERT INTO
    "system_prompt_template" (
        "id",
        "name",
        "category",
        "content",
        "sub_content",
        "required_vars",
        "is_default",
        "is_active"
    )
VALUES (
        6,
        '기본 QA Prompt',
        'qa',
        '당신은 친절하고 이해하기 쉬운 설명을 제공하는 AI 어시스턴트입니다.
사용자의 질문이나 요청에 대해 정확하고 간결하게, 그리고 가능하다면 추가적인 배경 지식과 예시를 곁들여 답변하세요.
어려운 용어나 개념이 나올 경우, 초보자도 이해할 수 있도록 쉽게 풀어 설명하고, 필요하다면 목록·표·코드 블록 등을 활용하세요.
대화는 항상 존중과 긍정적인 어조를 유지하며, 사용자의 의도와 목표를 먼저 확인한 뒤에 답변을 구성합니다.

응답 스타일 가이드:

[CONTEXT]가 존재하는 경우 : [Context]를 [USER_PROMPT]에 따라 응답합니다.
[CONTEXT]가 존재하지 않는 경우 : 사용자가 요청한 [USER_PROMPT]에 따라 응답합니다.

질문이 모호하면 추가 질문으로 의도 명확히 하기 사용자가 원할 경우 심화 정보 제공',
        '',
        '[]',
        1,
        1
    );

INSERT INTO
    "system_prompt_template" (
        "id",
        "name",
        "category",
        "content",
        "sub_content",
        "required_vars",
        "is_default",
        "is_active"
    )
VALUES (
        7,
        '출장 계획서',
        'doc_gen',
        '당신은 출장 계획서를 작성하는 AI 어시스턴트입니다.',
        '
아래 제공된 변수 값을 기반으로 출장 목적·일정·세부 계획을 명확하고 체계적으로 문서 형식으로 작성하세요.

작성 규칙:
1. 문서는 제목, 기본 정보(작성일·작성자·부서), 출장 개요, 세부 일정, 예상 경비, 기타 참고 사항 순서로 구성
2. 일정은 날짜와 시간, 활동 내용을 포함한 표 형식으로 제시
3. 불필요한 수식어나 장황한 표현 없이, 핵심 내용을 간결하게 서술
4. 지시어와 변수를 결합하여 자연스럽게 문장 작성
5. 금액과 날짜는 원 단위, YYYY-MM-DD 형식 유지
',
        '["제목", "작성일", "작성자", "부서"]',
        0,
        1
    );

INSERT INTO
    "system_prompt_template" (
        "id",
        "name",
        "category",
        "content",
        "sub_content",
        "required_vars",
        "is_default",
        "is_active"
    )
VALUES (
        8,
        '채용 공고',
        'doc_gen',
        '당신은 채용 공고 문서를 작성하는 AI 어시스턴트입니다.',
        ' 
아래 제공된 변수 값을 참고하여 회사와 채용 포지션 정보를 명확하고 매력적으로 문서 형식으로 작성하세요.

작성 규칙:
1. 문서는 직무명, 회사 소개, 담당 업무, 자격 요건, 우대 사항, 근무 조건, 지원 방법 순으로 구성
2. 중요 정보가 돋보이도록 불릿 포인트와 명확한 문장 사용
3. 불필요한 수식어나 반복 표현은 제거하며, 지원자 입장에서 이해하기 쉽게 작성
4. 제공된 변수들을 자연스럽게 연결하여 문장 작성
5. 날짜는 YYYY-MM-DD 형식으로 표기
',
        '["직무명", "회사소개", "담당업무", "자격요건", "우대사항", "근무조건", "지원마감일", "지원방법"]',
        1,
        1
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        1,
        'text',
        '제목',
        NULL,
        '문서 제목 (예: 2025년 9월 일본 출장 계획서)'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        2,
        'datetime',
        '작성일',
        NULL,
        '문서 작성일 (예: 2025-08-12)'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        3,
        'text',
        '작성자',
        NULL,
        '작성자 성명'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        4,
        'text',
        '부서',
        NULL,
        '작성자의 부서명'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        5,
        'text',
        '직무명',
        NULL,
        '채용하는 직무명 (예: 소프트웨어 엔지니어)'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        6,
        'text',
        '회사소개',
        NULL,
        '회사 개요 및 비전 소개'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        7,
        'text',
        '담당업무',
        NULL,
        '직무별 주요 업무 내용'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        8,
        'text',
        '자격요건',
        NULL,
        '필수 자격 및 능력 조건'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        9,
        'text',
        '우대사항',
        NULL,
        '우대하는 경험이나 기술 (선택 사항)'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        10,
        'text',
        '근무조건',
        NULL,
        '근무지, 근무 형태, 근무 시간 등 근무 관련 사항'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        11,
        'datetime',
        '지원마감일',
        NULL,
        '채용 지원 마감일'
    );

INSERT INTO
    "system_prompt_variables" (
        "id",
        "type",
        "key",
        "value",
        "description"
    )
VALUES (
        12,
        'text',
        '지원방법',
        NULL,
        '지원 절차나 연락처 등 안내'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        1,
        7,
        1,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        2,
        7,
        2,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        3,
        7,
        3,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        4,
        7,
        4,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        5,
        8,
        5,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        6,
        8,
        6,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        7,
        8,
        7,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        8,
        8,
        8,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        9,
        8,
        9,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        10,
        8,
        10,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        11,
        8,
        11,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping" (
        "id",
        "template_id",
        "variable_id",
        "created_at",
        "updated_at"
    )
VALUES (
        12,
        8,
        12,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "users" (
        "id",
        "role",
        "username",
        "name",
        "password",
        "department",
        "position",
        "pfp_filename",
        "bio",
        "daily_message_limit",
        "suspended",
        "security_level",
        "created_at",
        "updated_at",
        "expires_at"
    )
VALUES (
        1,
        'user',
        'ruah0807',
        '김루아',
        'ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f',
        'AI 연구소',
        '연구원',
        NULL,
        '',
        NULL,
        0,
        3,
        '2025-08-13 13:54:54',
        '2025-08-13 13:54:54',
        '2025-08-12 07:01:02'
    );

INSERT INTO
    "users" (
        "id",
        "role",
        "username",
        "name",
        "password",
        "department",
        "position",
        "pfp_filename",
        "bio",
        "daily_message_limit",
        "suspended",
        "security_level",
        "created_at",
        "updated_at",
        "expires_at"
    )
VALUES (
        2,
        'user',
        'rlwjd123',
        '조기정',
        'ee63c6506c68d4613b9553820393f22db66a1dbc9ba6dc5640df9fce741e6258',
        'AI 연구소',
        '선임 연구원',
        NULL,
        '',
        NULL,
        0,
        3,
        '2025-08-14 15:06:03',
        '2025-08-14 15:06:03',
        '2025-08-14 06:06:03'
    );

INSERT INTO
    "users" (
        "id",
        "role",
        "username",
        "name",
        "password",
        "department",
        "position",
        "pfp_filename",
        "bio",
        "daily_message_limit",
        "suspended",
        "security_level",
        "created_at",
        "updated_at",
        "expires_at"
    )
VALUES (
        3,
        'user',
        'mingue123',
        '강민규',
        '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소',
        '연구원',
        NULL,
        '',
        NULL,
        0,
        2,
        '2025-08-14 15:10:22',
        '2025-08-14 15:10:22',
        '2025-08-14 06:10:22'
    );

INSERT INTO
    "users" (
        "id",
        "role",
        "username",
        "name",
        "password",
        "department",
        "position",
        "pfp_filename",
        "bio",
        "daily_message_limit",
        "suspended",
        "security_level",
        "created_at",
        "updated_at",
        "expires_at"
    )
VALUES (
        4,
        'admin',
        'jongwha123',
        '김종화',
        '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소',
        '본부장',
        NULL,
        '',
        NULL,
        0,
        3,
        '2025-08-14 15:11:42',
        '2025-08-14 15:11:42',
        '2025-08-14 06:11:42'
    );

INSERT INTO
    "users" (
        "id",
        "role",
        "username",
        "name",
        "password",
        "department",
        "position",
        "pfp_filename",
        "bio",
        "daily_message_limit",
        "suspended",
        "security_level",
        "created_at",
        "updated_at",
        "expires_at"
    )
VALUES (
        5,
        'user',
        'iju1234',
        '마주이',
        '03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소',
        '선임 연구원',
        NULL,
        '',
        NULL,
        0,
        1,
        '2025-08-14 15:12:03',
        '2025-08-14 06:12:03',
        '2025-08-14 06:12:03'
    );

COMMIT;