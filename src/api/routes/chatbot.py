from fastapi import APIRouter
from src.models.qwen import load_qwen_instruct_7b

router = APIRouter(tags=["chatbot"], prefix="/chat")
model, tokenizer = load_qwen_instruct_7b("/home/ruah0807/Desktop/project/rag_llm/backend/models/Qwen2.5-7B-Instruct-1M")

@router.post("/")
def chat_endpoint(request: dict):
    # 입력 받아서 모델 추론 후 결과 반환
    ...