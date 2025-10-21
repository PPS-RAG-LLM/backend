"""채팅 메트릭 업데이트 로직"""
from typing import Dict, Any
from errors import BadRequestError
from repository.workspace_chat import update_chat_metrics

def update_reasoning_duration(
    user_id: int,
    chat_id: int, 
    reasoning_duration: float
) -> Dict[str, Any]:
    """
    프론트엔드에서 계산한 reasoning_duration을 DB에 업데이트
    
    Args:
        user_id: 사용자 ID
        chat_id: 채팅 ID
        reasoning_duration: 추론 시간 (초)
    
    Returns:
        업데이트 결과
    """
    if reasoning_duration < 0:
        raise BadRequestError("reasoning_duration must be non-negative")
    
    update_chat_metrics(chat_id, user_id, reasoning_duration)
    
    return {
        "success": True,
        "chat_id": chat_id,
        "reasoning_duration": reasoning_duration
    }