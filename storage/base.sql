INSERT INTO
    "users"
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
    "users"
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
    "users"
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
    "users"
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
    "users"
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
        '2025-08-14 15:12:03',
        '2025-08-14 06:12:03'
    );

INSERT INTO
    "llm_models"
VALUES (
        1,
        'huggingface',
        'gpt_oss_20b',
        0,
        './service/storage/model/local_gpt_oss_20b',
        'qa',
        '',
        'base',
        0,
        1,
        '2025-09-03 07:52:05',
        '2025-09-03 07:52:05'
    );

INSERT INTO
    "llm_models"
VALUES (
        2,
        'huggingface',
        'qwen_2.5_7b_instruct',
        0,
        './service/storage/model/Qwen2.5-7B-Instruct-1M',
        'qa',
        '',
        'base',
        1,
        1,
        '2025-09-03 07:52:05',
        '2025-09-03 07:52:05'
    );

INSERT INTO
    "system_prompt_template"
VALUES (
        1,
        '연구보고서서 요약',
        'summary',
        '다음은 연구보고서 원문입니다. 이 내용을 바탕으로 다음 항목을 간결하고 명확하게 요약해 주세요.

연구 목적: 연구를 수행한 이유와 필요성

연구 방법: 실험 또는 조사 설계, 자료 수집 및 분석 방식

주요 결과: 실험·분석을 통해 얻은 핵심 데이터와 발견 사항

의의와 결론: 결과가 시사하는 바와 결론

한계점 및 향후 과제: 연구의 제약 사항과 후속 연구 방향

형식: 항목별 불릿 포인트로 작성하며, 불필요한 수식어나 중복 표현은 최소화하고 전문 용어를 유지하되 내용은 쉽게 이해할 수 있도록 정리해 주세요.',
        '[]',
        0,
        1
    );

INSERT INTO
    "system_prompt_template"
VALUES (
        2,
        '회의록 요약',
        'summary',
        '다음은 회의록 원문입니다. 이 내용을 바탕으로 다음 항목을 간결하고 명확하게 요약해 주세요.

회의 목적: 회의를 진행한 이유와 주요 의제

핵심 논의 사항: 주요 토론 내용과 결론

결정 및 합의사항: 회의 중 확정된 사항

추가 논의 필요 사항: 미완결 과제나 추후 논의 필요 주제

실행 계획: 담당자, 마감 기한, 진행 방식

형식: 항목별로 불릿 포인트로 작성하며, 불필요한 수식어나 중복 표현은 제거하고 핵심 정보만 포함해주세요.',
        '[]',
        0,
        1
    );

INSERT INTO
    "system_prompt_template"
VALUES (
        5,
        '이메일 스레드 요약',
        'summary',
        '다음은 한 이메일 스레드의 전체 내용입니다. 메일의 순서와 맥락을 고려하여 다음 항목을 간결하고 명확하게 요약해 주세요.

대화 주제: 해당 이메일 스레드의 전체적인 주제와 목적

주요 논의 및 진행 상황: 주고받은 내용의 핵심 포인트와 경과

결정 사항: 스레드에서 합의되거나 확정된 사안

미해결/추가 논의 필요 사항: 아직 결론이 나지 않은 사안

다음 액션 아이템: 누가, 무엇을, 언제까지 해야 하는지

작성 지침:

시간 순서 흐름이 잘 드러나도록 핵심 사건을 정리

불필요한 인사말·잡담·중복 내용 제거

중요한 인물과 날짜, 수치 등은 그대로 유지

항목별 불릿 포인트 형식으로 간결하게 작성',
        '[]',
        1,
        1
    );

INSERT INTO
    "system_prompt_template"
VALUES (
        6,
        '기본 QA Prompt',
        'qa',
        '당신은 친절하고 이해하기 쉬운 설명을 제공하는 AI 어시스턴트입니다.
사용자의 질문이나 요청에 대해 정확하고 간결하게, 그리고 가능하다면 추가적인 배경 지식과 예시를 곁들여 답변하세요.
어려운 용어나 개념이 나올 경우, 초보자도 이해할 수 있도록 쉽게 풀어 설명하고, 필요하다면 목록·표·코드 블록 등을 활용하세요.
대화는 항상 존중과 긍정적인 어조를 유지하며, 사용자의 의도와 목표를 먼저 확인한 뒤에 답변을 구성합니다.

응답 스타일 가이드:

핵심 정보 먼저 제시

불필요하게 긴 문장 대신 명확한 구조

예시, 정의, 비교 설명 활용

질문이 모호하면 추가 질문으로 의도 명확히 하기

사용자가 원할 경우 심화 정보 제공',
        '[]',
        1,
        1
    );

INSERT INTO
    "system_prompt_template"
VALUES (
        7,
        '출장 계획서',
        'doc_gen',
        '
당신은 출장 계획서를 작성하는 AI 어시스턴트입니다.  
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
    "system_prompt_template"
VALUES (
        8,
        '채용 공고',
        'doc_gen',
        '
당신은 채용 공고 문서를 작성하는 AI 어시스턴트입니다.  
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
    "system_prompt_variables"
VALUES (
        1,
        'text',
        '제목',
        NULL,
        '문서 제목 (예: 2025년 9월 일본 출장 계획서)'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        2,
        'datetime',
        '작성일',
        NULL,
        '문서 작성일 (예: 2025-08-12)'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        3,
        'text',
        '작성자',
        NULL,
        '작성자 성명'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        4,
        'text',
        '부서',
        NULL,
        '작성자의 부서명'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        5,
        'text',
        '직무명',
        NULL,
        '채용하는 직무명 (예: 소프트웨어 엔지니어)'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        6,
        'text',
        '회사소개',
        NULL,
        '회사 개요 및 비전 소개'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        7,
        'text',
        '담당업무',
        NULL,
        '직무별 주요 업무 내용'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        8,
        'text',
        '자격요건',
        NULL,
        '필수 자격 및 능력 조건'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        9,
        'text',
        '우대사항',
        NULL,
        '우대하는 경험이나 기술 (선택 사항)'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        10,
        'text',
        '근무조건',
        NULL,
        '근무지, 근무 형태, 근무 시간 등 근무 관련 사항'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        11,
        'datetime',
        '지원마감일',
        NULL,
        '채용 지원 마감일'
    );

INSERT INTO
    "system_prompt_variables"
VALUES (
        12,
        'text',
        '지원방법',
        NULL,
        '지원 절차나 연락처 등 안내'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        1,
        7,
        1,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        2,
        7,
        2,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        3,
        7,
        3,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        4,
        7,
        4,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        5,
        8,
        5,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        6,
        8,
        6,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        7,
        8,
        7,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        8,
        8,
        8,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        9,
        8,
        9,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        10,
        8,
        10,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        11,
        8,
        11,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );

INSERT INTO
    "prompt_mapping"
VALUES (
        12,
        8,
        12,
        '2025-08-13 13:54:55',
        '2025-08-13 13:54:55'
    );