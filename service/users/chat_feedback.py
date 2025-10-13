from typing import Optional
import csv
import json
from pathlib import Path
from utils import logger
from errors import NotFoundError
from repository.chat_feedback import (
    get_chat_by_id,
    save_feedback_metadata,
    update_feedback_metadata,
    get_feedback_by_file_info,
    list_all_feedbacks as repo_list_feedbacks,
    update_feedback_to_chat_worksapce,
)

logger = logger(__name__)

# 파일 저장 경로
TRAIN_DATA_DIR = Path("storage/train_data")
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_HEADERS = ["ChunkContext", "Question", "Answer", "UserAnswer", "ModelName"]


def generate_feedback_filename(
    category: str,
    subcategory: Optional[str],
    prompt_id: Optional[int]=None
) -> str:
    """
    피드백 CSV 파일명 생성
    형식: feedback_{category}_{subcategory}_p{prompt_id}.csv
    """
    parts = ["feedback", category]
    
    if subcategory:
        parts.append(subcategory)
    
    prompt_part = f"p{prompt_id}" if prompt_id else "p0"
    parts.append(prompt_part)
    
    return "_".join(parts) + ".csv"


def append_to_csv(
    file_path: Path,
    chunk_context: str,
    question: str,
    answer: str,
    user_feedback: bool,
    model_name: str
) -> None:
    """CSV 파일에 피드백 데이터 append"""
    file_exists = file_path.exists()
    
    # 파일 lock을 위한 간단한 처리 (동시성 이슈 방지)
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 파일이 새로 생성되는 경우 헤더 작성
            if not file_exists:
                writer.writerow(CSV_HEADERS)
                logger.info(f"Created new CSV file: {file_path}")
            
            # 데이터 추가
            writer.writerow([
                chunk_context,
                question,
                answer,
                str(user_feedback).lower(),  # true/false
                model_name
            ])
            logger.info(f"Appended feedback to CSV: {file_path}")
    except Exception as e:
        logger.error(f"Failed to append to CSV {file_path}: {e}")
        raise


def extract_answer_from_response(response_json: str) -> str:
    """response JSON에서 실제 답변 텍스트 추출"""
    try:
        data = json.loads(response_json)
        return data.get("text", response_json)
    except Exception:
        return response_json


def extract_context_from_response(response_json: str) -> str:
    """response JSON에서 RAG context 추출 (있는 경우)"""
    try:
        data = json.loads(response_json)
        sources = data.get("sources", []) # sources에서 context를 추출할 수 있음
        if sources:
            return "\n\n" + "="*80 + "\n\n".join([s.get("text", "") for s in sources if s.get("text")])
        return ""
    except Exception:
        return ""


def save_chat_feedback(
    user_id: int,
    chat_id: int,
    like: bool,  # "true" or "false"
    category: str,
    model_name: str,
    prompt_id: Optional[int] = None,
    subcategory: Optional[str] = None,
) -> dict:
    """
    사용자 피드백을 CSV에 저장하고 DB에 메타데이터 기록
    """
    # 1. 채팅 메시지 조회
    chat = get_chat_by_id(chat_id, user_id)
    if not chat:
        raise NotFoundError(f"Chat message not found: chat_id={chat_id}")

    # 2. 이미 피드백이 제출된 겨우 중복 방지 + 추가
    if chat.get("feedback") is not None:
        logger.warning(f"이미 피드백이 제출된 채팅입니다. 중복 방지: chat_id={chat_id}")
        from errors import BadRequestError
        raise BadRequestError(f"이미 피드백이 제출된 채팅입니다. 중복 방지: chat_id={chat_id}")
    
    # 2. response에서 실제 답변 추출
    answer = extract_answer_from_response(chat["response"]) # response는 json으로 저장되어 있음.
    question = chat["prompt"]
    
    # logger.debug(f"[PRINT RESPONSE]\n{chat["response"]}")
    # 3. response에서 context추출 시도
    context = extract_context_from_response(chat["response"])
    
    # 4. 파일명 생성
    filename = generate_feedback_filename(category, subcategory, prompt_id)
    file_path = TRAIN_DATA_DIR / filename
    
    # 5. CSV에 append
    append_to_csv(
        file_path=file_path,
        chunk_context=context or "",
        question=question,
        answer=answer,
        user_feedback=like,
        model_name=model_name
    )

    
    # 6. DB에 메타데이터 저장 또는 업데이트
    existing = get_feedback_by_file_info(category, subcategory, prompt_id)
    
    if existing:
        logger.debug(f"존재하는 피드백입니다. 업데이트 합니다.: {existing}")
        # 기존 레코드 업데이트 (updated_at 갱신)
        update_feedback_metadata(existing["id"])
        feedback_id = existing["id"]
    else:
        logger.debug(f"새로운 피드백입니다. 저장합니다.: {existing}")
        # 새 레코드 생성
        feedback_id = save_feedback_metadata(
            category=category,
            subcategory=subcategory,
            filename=filename,
            file_path=str(file_path.resolve().relative_to(Path.cwd().resolve())),
            prompt_id=prompt_id
        )
        logger.debug(f"feedback_id: {feedback_id}")
        update_feedback_to_chat_worksapce(chat_id, feedback_id)

    
    # 7. CSV 행 수 계산 (데이터 내부의 줄바꿈 고려)
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        row_count = sum(1 for _ in reader) - 1  # 헤더 제외
    
    return {
        "feedback_id": feedback_id,
        "filename": filename,
        "file_path": str(file_path.resolve().relative_to(Path.cwd().resolve())),
        "row_count": row_count,
        "message": "Feedback saved successfully"
    }


def list_feedbacks(
    category: Optional[str] = None,
    prompt_id: Optional[int] = None
) -> list[dict]:
    """저장된 피드백 파일 목록 조회"""
    feedbacks = repo_list_feedbacks(category, prompt_id)
    
    # 각 파일의 행 수 추가 (데이터 내부의 줄바꿈 고려)
    for fb in feedbacks:
        file_path = Path(fb["file_path"])
        if file_path.exists():
            try:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    row_count = sum(1 for _ in reader) - 1  # 헤더 제외
                fb["row_count"] = row_count
            except Exception:
                fb["row_count"] = 0
        else:
            fb["row_count"] = 0
    
    return feedbacks


def get_feedback_file_path(feedback_id: int) -> Path:
    """피드백 ID로 파일 경로 조회"""
    feedbacks = repo_list_feedbacks()
    for fb in feedbacks:
        if fb["id"] == feedback_id:
            return Path(fb["file_path"])
    raise NotFoundError(f"Feedback file not found: id={feedback_id}")