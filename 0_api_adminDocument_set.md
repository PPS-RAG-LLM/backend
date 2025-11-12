```
/v1/admin/vector/settings
{
  "embeddingModel": "qwen3_0_6b",
  "searchType": "hybrid",
  "chunkSize": 512,
  "overlap": 64
}

"hybrid", "semantic", "bm25"


/v1/admin/vector/security-levels

{
  "maxLevel": 3,
  "levels": {
    "1": "@일반@공개",
    "2": "@연구@연봉@개인정보",
    "3": "@부정"
  }
}

{
  "maxLevel": 3,
  "levels": {
    "1": "@일반@공개",
    "2": "@연구@연봉",
    "3": "@부정@개인정보"
  }
}

{
  "maxLevel": 3,
  "levels": {
    "1": "@일반@공개",
    "2": "@연구@연봉@개인정보@부정",
    "3": ""
  }
}


{
  "question": "부정청탁에 관련된 내용용을 알려줘",
  "topK": 5,
  "securityLevel": 3,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen",
  "searchMode": "semantic"
}

{
  "question": "부정청탁에 관련된 내용용을 알려줘",
  "topK": 5,
  "securityLevel": 2,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen",
  "searchMode": "bm25"
}

``````
/v1/admin/vector/settings
{
  "embeddingModel": "qwen3_0_6b",
  "searchType": "hybrid",
  "chunkSize": 512,
  "overlap": 64
}

"hybrid", "semantic", "bm25"


/v1/admin/vector/security-levels

{
  "maxLevel": 3,
  "levels": {
    "1": "@일반@공개",
    "2": "@연구@연봉@개인정보",
    "3": "@부정"
  }
}


{
  "question": "부정청탁에 관련된 내용용을 알려줘",
  "topK": 5,
  "securityLevel": 1,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen",
  "searchMode": "semantic"
}

{
  "question": "부정청탁에 관련된 내용용을 알려줘",
  "topK": 5,
  "securityLevel": 2,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen",
  "searchMode": "bm25"
}

```