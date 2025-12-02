from pydantic import BaseModel, Field
from typing import Optional
from fastapi import APIRouter, Path, Query, Body, Depends
from fastapi.responses import FileResponse
from service.users.chat_feedback import (
    save_chat_feedback,
    list_feedbacks,
    get_feedback_file_path,
)
from utils import logger, validate_category_subcategory, validate_category
from errors import BadRequestError, NotFoundError
from utils.auth import get_user_id_from_cookie

logger = logger(__name__)

feedback_router = APIRouter(tags=["Chat Feedback"], prefix="/v1/workspace")


class ChatFeedbackRequest(BaseModel):
    """채팅 피드백 요청"""
    chatId: int = Field(..., description="채팅 ID")
    like: bool = Field(..., description="좋아요(true)/싫어요(false)")
    promptId: Optional[int] = Field(None, description="프롬프트 템플릿 ID (선택사항)")


@feedback_router.post(
    "/{slug}/chat/feedback",
    summary="채팅 피드백 저장 (좋아요/싫어요)"
)
def save_feedback_endpoint(
    validated_category: tuple[str, Optional[str]] = Depends(validate_category_subcategory),
    body: ChatFeedbackRequest = Body(..., description="피드백 요청 본문"),
    user_id: int = Depends(get_user_id_from_cookie),
):
    """
    사용자가 채팅 응답에 좋아요/싫어요를 누를 때 호출됩니다.
    피드백 데이터는 CSV 파일에 저장되며, 파인튜닝 학습 데이터로 사용됩니다.
    """
    category, subcategory = validated_category # 카테고리 검증
    logger.info(f"[save_feedback] user_id={user_id}, category={category}, subcategory={subcategory}")
    result = save_chat_feedback(
        user_id=user_id,
        chat_id=body.chatId,
        like=body.like,
        category=category,
        prompt_id=body.promptId,
        subcategory=subcategory,
    )
    return {
        "success": True,
        "data": result
    }


@feedback_router.get(
    "/chat/feedback/list",
    summary="저장된 피드백 파일 목록 조회"
)
def list_feedbacks_endpoint(
    validated_category: str = Depends(validate_category),
    prompt_id: Optional[int] = Query(None, description="프롬프트 ID 필터"),
):
    """
    저장된 피드백 CSV 파일 목록을 조회합니다.
    각 파일의 메타데이터와 누적된 피드백 개수를 확인할 수 있습니다.
    """
    category = validated_category
    logger.info(f"[list_feedbacks] category={category}, prompt_id={prompt_id}, feedbacks={feedbacks}")
    feedbacks = list_feedbacks(category=category, prompt_id=prompt_id)
    
    return {
        "success": True,
        "data": {
            "feedbacks": feedbacks,
            "total": len(feedbacks)
        }
    }


@feedback_router.get(
    "/chat/feedback/{feedback_id}/download",
    summary="피드백 CSV 파일 다운로드"
)
def download_feedback_endpoint(
    feedback_id: int = Path(..., description="피드백 파일 ID"),
):
    """
    저장된 피드백 CSV 파일을 다운로드합니다.
    파인튜닝 학습에 사용할 수 있습니다.
    """
    try:
        file_path = get_feedback_file_path(feedback_id)
        
        if not file_path.exists():
            raise NotFoundError(f"Feedback file not found: {file_path}")
        
        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=file_path.name
        )
    except NotFoundError as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to download feedback file: {e}")
        raise BadRequestError(f"Failed to download file: {str(e)}")