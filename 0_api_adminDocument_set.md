```
/v1/admin/vector/settings
{
  "embeddingModel": "qwen3_0_6b",
  "searchType": "hybrid"
}



/v1/admin/vector/security-levels
{
  "doc_gen": {
    "maxLevel": 3,
    "levels": {
      "1": "@일반@공개",
      "2": "@연구@연봉@개인정보",
      "3": "@부정"
    }
  },
  "summary": {
    "maxLevel": 3,
    "levels": {
      "1": "@일반@공개",
      "2": "@연구@연봉@부정",
      "3": "@개인정보"
    }
  },
  "qna": {
    "maxLevel": 3,
    "levels": {
      "1": "@일반@공개",
      "2": "@연구@연봉@부정@개인정보",
      "3": ""
    }
  }
}


{
  "chunkSize": 512,
  "overlap": 64,
  "taskTypes": [
    "doc_gen",
    "summary",
    "qna"
  ]
}


1
{
  "question": "부정청닥에 관련된 내용용을 알려줘.",
  "topK": 5,
  "securityLevel": 3,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen"
}
2
{
  "question": "부정청닥에 관련된 내용용을 알려줘.",
  "topK": 5,
  "securityLevel": 1,
  "sourceFilter": [
    "string"
  ],
  "taskType": "doc_gen"
}

```