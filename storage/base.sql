BEGIN;


INSERT INTO
    "users" (
        "id","role","username","name","password","department","position",
        "pfp_filename","bio","daily_message_limit","suspended","security_level",
        "created_at","updated_at","expires_at"
    )
VALUES 
    (
        1,
        'user','ruah0807','김루아','ef797c8118f02dfb649607dd5d3f8c7623048c9c063d532cc95c5ed7a898a64f',
        'AI 연구소','연구원',NULL,'',NULL,0,3,
        '2025-08-13 13:54:54','2025-08-13 13:54:54', NULL
    ),
     (
        2,'user','rlwjd123','조기정','ee63c6506c68d4613b9553820393f22db66a1dbc9ba6dc5640df9fce741e6258',
        'AI 연구소','선임 연구원',NULL,'',NULL,0,3,
        '2025-08-14 15:06:03','2025-08-14 15:06:03', NULL
    ),
    (
        3,'user','mingue123','강민규','03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소','연구원',NULL,'',NULL,0,2,
        '2025-08-14 15:10:22','2025-08-14 15:10:22', NULL
    ),
    (
        4,'admin','jongwha123','김종화','03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소','본부장',NULL,'', NULL,0,3,
        '2025-08-14 15:11:42','2025-08-14 15:11:42', NULL
    ),
    (
        5,'user','iju1234','마주이','03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4',
        'AI 연구소','선임 연구원',NULL,'',NULL,0,1,
        '2025-08-14 15:12:03','2025-08-14 06:12:03','2025-08-14 06:12:03'
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
VALUES (1,'huggingface','gpt-oss-20b',0,'./storage/model/gpt-oss-20b',
        'all','base',1,1,'2025-09-03 07:52:05','2025-09-03 07:52:05'),
        (2,'huggingface','Qwen2.5-7B-Instruct-1M',0,'./storage/model/Qwen2.5-7B-Instruct-1M',
        'all','base',0,1,'2025-09-03 07:52:05','2025-09-03 07:52:05'),
        (3,'huggingface','Qwen3-8B',0,'./storage/model/Qwen3-8B',
        'all','base',0,1,'2025-09-03 07:52:05','2025-09-03 07:52:05'),
        (4,'huggingface','Qwen3-Omni-30B-A3B-Instruct',0,'./storage/model/Qwen3-Omni-30B-A3B-Instruct',
        'qa','huggingface',0,1,'2025-09-03 07:52:05','2025-09-03 07:52:05'),
        (5,'huggingface','Qwen3-14B',0,'./storage/model/Qwen3-14B',
        'all','huggingface',0,1,'2025-09-03 07:52:05','2025-09-03 07:52:05');
        

INSERT INTO
    "rag_settings" (
        "id",
        "search_type",
        "chunk_size",
        "overlap",
        "embedding_key",
        "updated_at"
    )
VALUES (1,'hybrid',512,64,'embedding_qwen3_0_6b','2025-08-13 13:54:55');

-- (선택) doc_gen 기본값을 하나로 제한하는 인덱스
CREATE UNIQUE INDEX IF NOT EXISTS uq_system_prompt_doc_gen_default
    ON "system_prompt_template" ("category", "name")
    WHERE "category"='doc_gen' AND "is_default" = 1;

INSERT INTO
    "system_prompt_template" (
        "id",
        "name",
        "category",
        "system_prompt",
        "user_prompt",
        "is_default",
        "is_active"
    )
VALUES 
    (80, 'qa_prompt', 'qa',
        '당신은 첫번째 친절하고 이해하기 쉬운 설명을 제공하는 AI 어시스턴트입니다.
        사용자의 질문이나 요청에 대해 정확하고 간결하게, 그리고 가능하다면 추가적인 배경 지식과 예시를 곁들여 답변하세요.
        어려운 용어나 개념이 나올 경우, 초보자도 이해할 수 있도록 쉽게 풀어 설명하고, 필요하다면 목록·표·코드 블록 등을 활용하세요.
        대화는 항상 존중과 긍정적인 어조를 유지하며, 사용자의 의도와 목표를 먼저 확인한 뒤에 답변을 구성합니다.

        응답 스타일 가이드:

        <context> 또는 <document>가 존재하는 경우 : <context> 또는 <document>를 <user_prompt>에 따라 응답합니다.
        <context> 또는 <document>가 존재하지 않는 경우 : 사용자가 요청한 <user_prompt>에 따라 응답합니다.

        질문이 모호하면 추가 질문으로 의도 명확히 하기 사용자가 원할 경우 심화 정보 제공',
        '',0,1),
    (81, 'qa_prompt', 'qa',
        'You are the second AI assistant, designed to provide kind and easy-to-understand explanations.
        Respond to the user’s questions or requests accurately and concisely, and whenever possible, include helpful background knowledge and examples.
        If technical terms or complex concepts appear, explain them in simple language that beginners can understand, using lists, tables, or code blocks when appropriate.
        Always maintain a respectful and positive tone, and make sure to identify the user’s intent and goal first before composing your answer.

        ### Response Style Guide
        - When a <context> or <document> exists: respond to the user’s <user_prompt> based on that <context> or <document>.
        - When no <context> or <document> exists: respond directly according to the user’s <user_prompt>.
        - If a question is ambiguous, ask clarifying questions to understand the intent.
        - If the user wants it, provide more in-depth or advanced information.',
        '',1,1),
    (90,'summary_prompt','summary',
        '당신은 요약 전문 AI 어시스턴트 입니다. 사용자가 요청한 <context> 또는 <document>를 <user_prompt>에 따라 요약하세요.',
        '',0,1),
    (91,'summary_prompt','summary',
        'You are a professional summarizing assistant. Summarize the user’s <context> or <document> based on their <user_prompt>.',
        '',1,1),

-- doc_gen 카테고리 템플릿 8건 (business_trip 2, meeting 3, 보고서 3)
    (101, 'business_trip', 'doc_gen',
     '당신은 기업 출장 계획서를 작성하는 AI 어시스턴트입니다. [출장제목], [출장기간], [출장목적] 정보를 활용해 체계적인 문서를 작성하세요.',
     '문서는 개요, 일정, 예산, 준비물 섹션을 포함합니다.',
      1, 1),
    (102, 'business_trip', 'doc_gen',
     '당신은 간단한 출장 계획 메모를 작성하는 도우미입니다. 입력된 변수만 사용해서 핵심 일정과 준비물을 요약하세요.',
     '표 형식 일정과 준비물 체크리스트를 포함하세요.',
      0, 1),
    (103, 'meeting', 'doc_gen',
     '당신은 공식 회의록을 작성하는 AI 비서입니다. [회의명], [회의일시], [참석자], [주요논의] 항목을 활용해 명확한 회의록을 만드세요.',
     '요약, 상세 논의, 결정 사항 섹션을 포함하세요.',
      1, 1),
    (104, 'meeting', 'doc_gen',
     '당신은 의사결정에 초점을 둔 회의록을 작성합니다. 제공된 정보를 바탕으로 핵심 결론만 정리하세요.',
     '결정 사항과 근거를 bullet로 작성하세요.',
      0, 1),
    (105, 'meeting', 'doc_gen',
     '당신은 후속 조치 중심의 회의노트를 작성하는 도우미입니다. 후속 일정과 담당자를 명확히 정리하세요.',
     '액션 아이템 표를 포함하세요.',
      0, 1),
    (106, 'report', 'doc_gen',
     '당신은 주간 업무 보고서를 작성하는 AI 비서입니다. [보고제목], [보고기간], [성과요약], [주요지표]를 활용해 구조화된 보고서를 작성하세요.',
     '성과 요약, 지표 표, 향후 계획을 포함하세요.',
      1, 1),
    (107, 'report', 'doc_gen',
     '당신은 리스크 중심 보고서를 작성합니다. 제공된 정보를 토대로 위험과 대응 방안을 명확히 정리하세요.',
     '리스크와 대응 방안을 표로 작성하세요.',
      0, 1),
    (108, 'report', 'doc_gen',
     '당신은 이슈 보고서를 작성하는 도우미입니다. 발생한 이슈와 요청 사항을 명확히 정리해 관리자에게 전달하세요.',
     '요약, 상세 이슈, 요청 사항 섹션을 포함하세요.',
      0, 1);

-- 프롬프트에 필요한 변수 정의
INSERT INTO
"system_prompt_variables" ("id","type","required","key","value","description")
VALUES
(201,'text', 0, '출장제목', '뉴욕 해외 파트너십 체결 출장', '출장 문서의 제목'),
(202,'start_date', 1, '시작일', '2025-10-03', '시작일시'),
(203,'end_date', 1, '종료일', '2025-10-12', '종료일시'),
(204,'text', 0, '출장목적', '해외 파트너와 신규 공급 계약 협의', '출장의 목적 및 기대 효과'),
(205,'text', 0, '주요일정', '10/04 파트너사 미팅 · 10/06 공장 실사 · 10/08 협상 총괄 회의', '주요 일정 요약'),
(206,'text', 0, '준비물', '계약서 초안, 제품 샘플, 프레젠테이션 자료', '준비해야 할 물품 목록'),
(207,'text', 0, '결재자', '홍길동 전략사업본부장', '결재를 맡은 책임자'),
(208,'text', 0, '회의명', '북미 시장 진출 전략 회의', '회의의 제목 또는 안건'),
(209,'datetime', 1, '회의일시', '2025-10-05T14:00:00', '회의가 진행된 일시'),
(210,'text', 0, '참석자', '박유진, 김태호, Annie Smith, John Park', '참석자 명단'),
(211,'text', 0, '주요논의', '미국 시장 론칭 일정과 현지 마케팅 전략 수립', '핵심 논의 내용 요약'),
(212,'text', 0, '결정사항', '11월 내 계약 체결, 초기 물량 5,000대 공급 확정', '회의에서 확정된 결정 사항'),
(213,'text', 0, '후속조치', '각 부서별 실행 계획서 10/15까지 제출', '회의 후 수행해야 할 조치'),
(214,'text', 0, '발표자료', '시장 조사 리포트, 제품 로드맵 슬라이드', '공유된 발표 자료 목록'),
(215,'text', 0, '참고자료', '2024 글로벌 실적 보고서, 경쟁사 분석', '회의 참고 자료'),
(216,'text', 0, '담당자', '박유진 해외사업팀장', '후속 조치 담당자'),
(217,'text', 0, '보고제목', '2025년 4분기 미국 시장 진출 계획 보고', '보고서 제목'),
(219,'text', 0, '성과요약', '현지 파트너 3곳과 사전 MOU 체결 완료', '핵심 성과 요약'),
(220,'text', 0, '주요지표', '신규 리드 45건, 예상 매출 12억원, 고객 만족도 92점', '성과를 보여주는 주요 지표'),
(221,'text', 0, '리스크', 'FDA 인증 지연 가능성 및 환율 변동', '인식된 리스크 요약'),
(222,'text', 0, '대응방안', '인증 전문 대행사 선정 및 환헤지 전략 검토', '리스크 대응 계획'),
(223,'text', 0, '이슈', '물류 창고 확보 지연으로 일정 재조정 필요', '발생한 주요 이슈'),
(224,'text', 0, '요청사항', '마케팅 예산 3억원 추가 편성 승인 요청', '추가 요청 또는 필요 지원'),
(225,'text', 0, '작성부서', '해외사업개발팀', '보고서를 작성한 부서'),
(226,'textarea', 1, '요청사항', '파트너 계약 마무리를 위한 법무 검토 지원이 필요합니다.', '요청사항을 입력하세요');

-- 템플릿과 변수 매핑
INSERT INTO
    "prompt_mapping" ("id","template_id","variable_id","created_at","updated_at")
VALUES
    (301,101,201,'2025-09-01 09:00:00','2025-09-01 09:00:00'),
    (302,101,202,'2025-09-01 09:00:00','2025-09-01 09:00:00'),
    (303,101,203,'2025-09-01 09:00:00','2025-09-01 09:00:00'),
    (304,102,202,'2025-09-01 09:05:00','2025-09-01 09:05:00'),
    (305,102,203,'2025-09-01 09:05:00','2025-09-01 09:05:00'),
    (306,102,204,'2025-09-01 09:05:00','2025-09-01 09:05:00'),
    (307,102,205,'2025-09-01 09:05:00','2025-09-01 09:05:00'),
    (308,103,207,'2025-09-01 09:10:00','2025-09-01 09:10:00'),
    (309,103,208,'2025-09-01 09:10:00','2025-09-01 09:10:00'),
    (310,103,209,'2025-09-01 09:10:00','2025-09-01 09:10:00'),
    (311,103,210,'2025-09-01 09:10:00','2025-09-01 09:10:00'),
    (312,104,207,'2025-09-01 09:15:00','2025-09-01 09:15:00'),
    (313,104,208,'2025-09-01 09:15:00','2025-09-01 09:15:00'),
    (314,104,209,'2025-09-01 09:15:00','2025-09-01 09:15:00'),
    (315,104,211,'2025-09-01 09:15:00','2025-09-01 09:15:00'),
    (316,105,207,'2025-09-01 09:20:00','2025-09-01 09:20:00'),
    (317,105,212,'2025-09-01 09:20:00','2025-09-01 09:20:00'),
    (318,105,215,'2025-09-01 09:20:00','2025-09-01 09:20:00'),
    (319,106,216,'2025-09-01 09:25:00','2025-09-01 09:25:00'),
    (320,106,217,'2025-09-01 09:25:00','2025-09-01 09:25:00'),
    (321,106,202,'2025-09-01 09:25:00','2025-09-01 09:25:00'),
    (322,106,203,'2025-09-01 09:25:00','2025-09-01 09:25:00'),
    (323,106,219,'2025-09-01 09:25:00','2025-09-01 09:25:00'),
    (324,107,216,'2025-09-01 09:30:00','2025-09-01 09:30:00'),
    (325,107,220,'2025-09-01 09:30:00','2025-09-01 09:30:00'),
    (326,107,221,'2025-09-01 09:30:00','2025-09-01 09:30:00'),
    (327,108,216,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (328,108,222,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (329,108,223,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (330,108,224,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (331,108,225,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (332,101,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (333,102,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (334,103,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (335,104,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (336,105,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (337,106,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (338,107,226,'2025-09-01 09:35:00','2025-09-01 09:35:00'),
    (339,108,226,'2025-09-01 09:35:00','2025-09-01 09:35:00');


-- doc_gen 프롬프트와 LLM 매핑
INSERT INTO
    "llm_prompt_mapping" (
        "id",
        "llm_id",
        "prompt_id",
        "rouge_score",
        "response",
        "created_at",
        "updated_at"
    )
VALUES
    (401, 1, 101, 0.82,
     'llm응답 예시-출장 계획서 V1: 일정과 예산이 균형 있게 작성되었습니다.',
     '2025-09-01 10:00:00', '2025-09-01 10:00:00'),
    (402, 2, 101, 0.78,
     'llm응답 예시-Qwen 버전의 출장 계획서 초안입니다. 일정과 준비물이 잘 정리되었습니다.',
     '2025-09-01 10:05:00', '2025-09-01 10:05:00'),
    (403, 1, 102, 0.74,
     'llm응답 예시-간단 출장 메모 초안입니다. 준비물 체크리스트를 업데이트하세요.',
     '2025-09-01 10:10:00', '2025-09-01 10:10:00'),
    (404, 1, 103, 0.90,
     'llm응답 예시-회의록 버전1: 논의 요약과 결정 사항이 명확합니다.',
     '2025-09-01 10:15:00', '2025-09-01 10:15:00'),
    (405, 2, 104, 0.86,
     'llm응답 예시-의사결정 중심 회의록입니다. 결정 사항이 간결하게 정리되었습니다.',
     '2025-09-01 10:20:00', '2025-09-01 10:20:00'),
    (406, 2, 105, 0.88,
     'llm응답 예시-후속 조치 중심 회의노트 버전입니다. 담당자와 기한을 확인하세요.',
     '2025-09-01 10:25:00', '2025-09-01 10:25:00'),
    (407, 1, 106, 0.91,
     'llm응답 예시-주간 업무 보고서 초안입니다. 성과 지표를 표로 정리했습니다.',
     '2025-09-01 10:30:00', '2025-09-01 10:30:00'),
    (408, 2, 107, 0.83,
     'llm응답 예시-리스크 보고서 버전입니다. 리스크별 대응 방안을 검토하세요.',
     '2025-09-01 10:35:00', '2025-09-01 10:35:00'),
    (409, 1, 108, 0.80,
     'llm응답 예시-이슈 보고서 초안입니다. 요청 사항을 재확인하세요.',
     '2025-09-01 10:40:00', '2025-09-01 10:40:00');

-- fine_tune_datasets 테이블 예시 데이터
INSERT INTO
    "fine_tune_datasets" (
        "id",
        "name", 
        "category",
        "prompt_id",
        "path",
        "record_count",
        "created_at",
        "updated_at"
    )
VALUES 
    -- doc_gen 카테고리 데이터셋 (각 프롬프트별로)
    (1, 'business_trip_dataset_v1', 'doc_gen', 101, './storage/train_data/business_trip_v1.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (2, 'business_trip_dataset_v2', 'doc_gen', 102, './storage/train_data/business_trip_v2.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (3, 'meeting_formal_dataset', 'doc_gen', 103, './storage/train_data/meeting_formal.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (4, 'meeting_decision_dataset', 'doc_gen', 104, './storage/train_data/meeting_decision.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (5, 'meeting_action_dataset', 'doc_gen', 105, './storage/train_data/meeting_action.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (6, 'report_weekly_dataset', 'doc_gen', 106, './storage/train_data/report_weekly.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (7, 'report_risk_dataset', 'doc_gen', 107, './storage/train_data/report_risk.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    (8, 'report_issue_dataset', 'doc_gen', 108, './storage/train_data/report_issue.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    -- summary 카테고리 데이터셋 (프롬프트 ID 5)
    (9, 'summary_general_dataset', 'summary', 91, './storage/train_data/summary_general.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00'),
    -- qa 카테고리 데이터셋 (프롬프트 ID 100)
    (10, 'qa_general_dataset', 'qa', 80, './storage/train_data/data.csv', 0, '2025-09-25 09:00:00', '2025-09-25 09:00:00');

-- llm_eval_runs mock rows (pdf_list 포함)
INSERT INTO "llm_eval_runs" (
  "id","mapping_id","llm_id","prompt_id","category","subcategory","model_name",
  "prompt_text","user_prompt","rag_refs","answer_text","acc_score","meta","created_at","pdf_list"
) VALUES
  (601, 401, 1, 101, 'doc_gen', 'business_trip', 'gpt-oss-20b',
   '당신은 기업 출장 계획서를 작성하는 AI 어시스턴트입니다. [출장제목], [출장기간], [출장목적] 정보를 활용해 체계적인 문서를 작성하세요.\n다음 주 일본 도쿄 출장 계획서를 작성해주세요. 기간은 2025년 10월 15일부터 17일까지이며, 목적은 파트너사 미팅입니다.',
   '다음 주 일본 도쿄 출장 계획서를 작성해주세요. 기간은 2025년 10월 15일부터 17일까지이며, 목적은 파트너사 미팅입니다.',
   '["milvus://test_20250924_123456/doc_tokyo_trip", "file://business_travel_policy.pdf"]',
   '# 일본 도쿄 출장 계획서\n\n## 출장 개요...\n(생략)',
   78.5, '{"tokens_input":95,"tokens_output":187,"latency_ms":2340}',
   '2025-09-24 14:30:15',
   '["business_travel_policy.pdf"]'
  ),

  (602, 402, 2, 102, 'doc_gen', 'business_trip', 'Qwen2.5-7B-Instruct-1M',
   '당신은 기업 출장 계획서를 작성하는 AI 어시스턴트입니다. [출장제목], [출장기간], [출장목적] 정보를 활용해 체계적인 문서를 작성하세요.\n부산 지사 방문 계획서를 작성해주세요. 기간은 2025년 10월 20일부터 21일까지이며, 목적은 분기 실적 점검입니다.',
   '부산 지사 방문 계획서를 작성해주세요. 기간은 2025년 10월 20일부터 21일까지이며, 목적은 분기 실적 점검입니다.',
   '["milvus://test_20250924_123456/doc_busan_visit", "file://quarterly_review_template.pdf"]',
   '# 부산 지사 방문 계획서\n\n## 출장 개요...\n(생략)',
   82.3, '{"tokens_input":98,"tokens_output":165,"latency_ms":1890}',
   '2025-09-24 15:15:22',
   '["quarterly_review_template.pdf"]'
  ),

  (603, 404, 1, 103, 'doc_gen', 'meeting', 'gpt-oss-20b',
   '당신은 공식 회의록을 작성하는 AI 비서입니다. [회의명], [회의일시], [참석자], [주요논의] 항목을 활용해 명확한 회의록을 만드세요.\n2025년 9월 기획회의 회의록을 작성해주세요. 참석자는 김팀장, 박과장, 이대리이며, 신제품 출시 일정에 대해 논의했습니다.',
   '2025년 9월 기획회의 회의록을 작성해주세요. 참석자는 김팀장, 박과장, 이대리이며, 신제품 출시 일정에 대해 논의했습니다.',
   '["milvus://test_20250924_123456/doc_meeting_sept", "file://product_launch_schedule.pdf"]',
   '# 2025년 9월 기획회의 회의록\n\n## 회의 개요...\n(생략)',
   85.7, '{"tokens_input":112,"tokens_output":203,"latency_ms":2650}',
   '2025-09-24 16:45:33',
   '["product_launch_schedule.pdf"]'
  ),

  (604, 407, 1, 106, 'doc_gen', 'report', 'gpt-oss-20b',
   '당신은 주간 업무 보고서를 작성하는 AI 비서입니다. [보고제목], [보고기간], [성과요약], [주요지표]를 활용해 구조화된 보고서를 작성하세요.\n9월 3주차 개발팀 주간 보고서를 작성해주세요. 이번 주 주요 성과는 API 개발 완료와 테스트 커버리지 80% 달성입니다.',
   '9월 3주차 개발팀 주간 보고서를 작성해주세요. 이번 주 주요 성과는 API 개발 완료와 테스트 커버리지 80% 달성입니다.',
   '["milvus://test_20250924_123456/doc_weekly_dev", "file://dev_team_metrics.pdf"]',
   '# 개발팀 주간 보고서 (9월 3주차)\n\n## 보고 개요...\n(생략)',
   79.2, '{"tokens_input":128,"tokens_output":245,"latency_ms":3120}',
   '2025-09-24 17:20:45',
   '["dev_team_metrics.pdf"]'
  ),

  (605, NULL, 2, 90, 'summary', '요약 프롬프트', 'Qwen2.5-7B-Instruct-1M',
   '당신은 요약 전문 AI 어시스턴트 입니다. 사용자가 요청한 [Context]를 [USER_PROMPT]에 따라 요약하세요.\n다음 회의록을 3줄로 요약해주세요: ...',
   '다음 회의록을 3줄로 요약해주세요: ...',
   '["milvus://test_20250924_789012/doc_half_year_review"]',
   '2025년 상반기 실적 검토 결과 매출 목표 달성률 105%를 달성했습니다... (생략)',
   73.8, '{"tokens_input":156,"tokens_output":64,"latency_ms":1540}',
   '2025-09-24 18:10:12',
   '["half_year_review_minutes.pdf"]'
  ),

  (606, NULL, 1, 80, 'qa', 'QA Prompt', 'gpt-oss-20b',
   '당신은 친절하고 이해하기 쉬운 설명을 제공하는 AI 어시스턴트입니다. ...\n회사 휴가 정책에 대해 알려주세요. 연차는 몇 일까지 사용할 수 있나요?',
   '회사 휴가 정책에 대해 알려주세요. 연차는 몇 일까지 사용할 수 있나요?',
   '["milvus://test_20250924_789012/doc_vacation_policy", "file://hr_manual_2025.pdf"]',
   '회사 휴가 정책에 대해 안내드립니다.\n\n## 연차 휴가...\n(생략)',
   71.4, '{"tokens_input":142,"tokens_output":198,"latency_ms":2890}',
   '2025-09-24 19:05:28',
   '["hr_manual_2025.pdf"]'
  );


COMMIT;