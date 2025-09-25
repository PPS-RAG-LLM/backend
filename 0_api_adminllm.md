==/v1/test/llm/runs/ensure==

{
  "category": "doc_gen",
  "subcategory": "business_trip",
  "modelName": "Qwen3-8B",
  "userPrompt": "부산 지사 방문 계획서를 작성해주세요. 기간은 2025년 10월 20일부터 21일까지이며, 목적은 분기 실적 점검입니다.",
  "promptId": 101,
  "pdfList": [
    "string"
  ],
  "max_tokens": 512,
  "temperature": 0.7
}


====

/v1/admin/llm/model/insert-base
{
  "name": "Qwen2.5-7B-Instruct-1M",
  "model_path": "storage/model/Qwen2.5-7B-Instruct-1M",
  "provider": "huggingface",
  "category": "all"
}

{
  "name": "gpt-oss-20b",
  "model_path": "storage/model/gpt-oss-20b",
  "provider": "huggingface",
  "category": "all"
}


# 프롬프트 추가 /v1/admin/llm/prompts

{
  "title": "출장계획서_V1",
  "prompt": "일정: {{일정}}\n작성자: {{작성자}}\n출장지: {{출장지}}\n요청: {{내용 및 요청 사항}}\n위 정보를 바탕으로 ...",
  "variables": [
    {"key":"일정", "value":"2025.08.25 ~ 2025.09.25", "type":"text"},
    {"key":"작성자", "value":"홍길동", "type":"text"},
    {"key":"출장지", "value":"서울 광진구", "type":"text"},
    {"key":"내용 및 요청 사항", "value":"texttext...", "type":"text"}
  ]
}

# 프롬프트 변경
{ "category":"doc_gen", "subtask":"출장계획서", "promptId": 123 }

# 파인튜닝 테스트
{
  "category": "qa",
  "subcategory": "qa",
  "baseModelName": "Qwen2.5-7B-Instruct-1M",
  "saveModelName": "Qwen2.5-7B-Instruct-1M-질의응답",
  "systemPrompt": "위 글을 참고하여 대답해 주세요",
  "batchSize": 4,
  "epochs": 3,
  "learningRate": 0.0002,
  "overfittingPrevention": true,
  "trainSetFile": "/home/work/CoreIQ/backend/storage/train_data/data.csv",
  "gradientAccumulationSteps": 8,
  "quantizationBits": 8,
  "tuningType": "QLORA"
}

{
  "category": "summary",
  "subcategory": "summary",
  "baseModelName": "gpt-oss-20b",
  "saveModelName": "gpt_oss_20b_summary-요약",
  "systemPrompt": "위 글을 참고하여 요약해 주세요",
  "batchSize": 4,
  "epochs": 3,
  "learningRate": 0.0002,
  "overfittingPrevention": true,
  "trainSetFile": "/home/work/CoreIQ/backend/storage/train_data/data.csv",
  "gradientAccumulationSteps": 8,
  "quantizationBits": 4,
  "tuningType": "QLORA"
}


=====================================================================================
POST	/v1/admin/llm/settings	TOPK 조정 – RAG 반환 문서 수 변경	
📤 요청 본문
{
  "embeddingModel": "string",
  "searchType": "hybrid",
  "chunkSize": 128,
  "overlap": 64
}

📨 응답
✅ 성공 응답
200 OK
{
  "success": true
}
GET	/v1/admin/llm/settings/model-list	모델 목록 조회	
GET /v1/admin/llm/settings/model-list
모델 목록 조회

📥 파라미터
• category (query) (string) [필수]
    

PUT	/v1/admin/llm/settings/model-load	모델 로드/언로드	application/json
{
  "modelName": "Qwen-3.3-summary-v1"
}

📨 응답
✅ 성공 응답
200 OK
{
  "success": true,
  "message": "모델 로드/언로드 완료"
}
GET	/v1/admin/llm/compare-models	json 저장되어있는 이전 모델별 테스트 결과 최신순 3가지 가져오기(사용자가 테스트 제출한 것 포함)	application/json
{
  "category": "summary",
  "modelId": 5,
  "promptId": 4,
  "prompt": "지정된 요약 프롬프트"
}

📨 응답
✅ 성공 응답
200 OK
{
  "modelList": [
    {
      "modelId": 5,
      "modelName": "Qwen-3.3-summary-v3",
      "answer": "",
      "rougeScore": 70
    },
    {
      "modelId": 4,
      "modelName": "Qwen-3.3-summary-v2",
      "answer": "",
      "rougeScore": 20
    },
    {
      "modelId": 3,
      "modelName": "Qwen-3.3-summary-v1",
      "answer": "",
      "rougeScore": 22
    }
  ]
}
GET	/v1/admin/llm/prompts	모델 프롬프트 카테고리별 목록 간단 조회	
GET /v1/admin/llm/prompts
모델 프롬프트 카테고리별 목록 간단 조회

📥 파라미터
• category (query) (string) [필수]
    

📨 응답
✅ 성공 응답
200 OK
{
  "category": "doc_summary",
  "promptList": [
    {
      "promptId": 7,
      "title": "회의 요약",
      "prompt": "당신은 회의 요약 전문가입니다. 회의 내용을 요약해주세요."
    },
    {
      "promptId": 8,
      "title": "출장 보고서 요약",
      "prompt": "출장보고서를 이와 같은 형식으로 요약해주세요. 출장 날짜, 출장 장소, 출장 목적, 출장 결과 등을 포함해주세요."
    }
  ]
}
POST	/v1/admin/llm/prompts	모델 프롬프트 생성	• category (query) (string) [필수]
    

📨 응답
✅ 성공 응답
200 OK
{
  "category": "doc_summary",
  "promptList": [
    {
      "promptId": 7,
      "title": "회의 요약",
      "prompt": "당신은 회의 요약 전문가입니다. 회의 내용을 요약해주세요."
    },
    {
      "promptId": 8,
      "title": "출장 보고서 요약",
      "prompt": "출장보고서를 이와 같은 형식으로 요약해주세요. 출장 날짜, 출장 장소, 출장 목적, 출장 결과 등을 포함해주세요."
    }
  ]
}
GET	/v1/admin/llm/prompt/{prompt_id}	모델 프롬프트 조회	모델 프롬프트 조회

📥 파라미터
• category (query) (string) [필수]
    
• promptId (path) (integer) [필수]
    

📨 응답
✅ 성공 응답
200 OK
{
  "promptId": 4,
  "title": "회의 요약",
  "prompt": "위 글을 참고하여 요약해서 사용자 질문에 답하여 주세요",
  "variables": [
    {
      "key": "date",
      "value": "2025-01-01",
      "type": "date-time"
    },
    {
      "key": "location",
      "value": "출장지를 입력하세요",
      "type": "string"
    }
  ]
}
PUT	/v1/admin/llm/prompt/{prompt_id}	모델 프롬프트 수정	모델 프롬프트 수정

📥 파라미터
• category (query) (string) [필수]
    
• promptId (path) (integer) [필수]
    

📤 요청 본문
application/json
{
  "title": "회의 요약",
  "prompt": "위 글을 참고하여 요약해서 사용자 질문에 답하여 주세요",
  "variables": [
    {
      "key": "date",
      "value": "2025-01-01",
      "type": "date-time"
    },
    {
      "key": "location",
      "value": "출장지를 입력하세요",
      "type": "string"
    }
  ]
}

📨 응답
✅ 성공 응답
200 OK
{
  "success": true
}
DELETE	/v1/admin/llm/prompt/{prompt_id}	모델 프롬프트 삭제	모델 프롬프트 삭제

📥 파라미터
• category (query) (string) [필수]
    
• promptId (path) (integer) [필수]
    

📨 응답
✅ 성공 응답
200 OK
{
  "success": true
}

POST	/v1/admin/llm/prompt/{prompt_id}	모델 프롬프트 관리 - 테스트 실행	모델 프롬프트 관리 - 테스트 실행

📥 파라미터
• category (query) (string) [필수]
    
• promptId (path) (integer) [필수]
    

📤 요청 본문
application/json
{
  "modelId": 5,
  "CoT": true,
  "prompt": "당신은 회의 요약 전문가입니다. 회의 내용을 요약해주세요.",
  "variables": []
}

📨 응답
✅ 성공 응답
200 OK
{
  "answer": "요약된 회의내용"
}