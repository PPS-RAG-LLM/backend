from pydantic import BaseModel, Field
from typing import Dict, Any

# 1. 일반적인 에러 (400, 401, 409, 422 등)
# Handler: base_api_exception_handler (기본)
class ErrorResponseDetail(BaseModel):
    detail: str = Field(..., description="에러 상세 메시지")

# 2. 권한 없음 (403)
# Handler: ForbiddenError
class ErrorResponseMessage(BaseModel):
    message: str = Field(..., description="권한 없음 메시지")

# 3. 실패/서버 에러 (404, 500)
# Handler: NotFoundError, InternalServerError
class ErrorResponseSuccessFalse(BaseModel):
    success: bool = Field(False, description="성공 여부 (항상 False)")
    error: str = Field(..., description="에러 메시지")

# --- 편의를 위한 공통 응답 정의 맵 ---
# 라우터에서 responses={**COMMON_ERRORS} 형태로 사용 가능
COMMON_ERRORS: Dict[int, Dict[str, Any]] = {
    400: {"model": ErrorResponseDetail, "description": "잘못된 요청 (Bad Request)"},
    401: {"model": ErrorResponseDetail, "description": "인증 필요 (Unauthorized)"},
    403: {"model": ErrorResponseMessage, "description": "권한 없음 (Forbidden)"},
    404: {"model": ErrorResponseSuccessFalse, "description": "리소스를 찾을 수 없음 (Not Found)"},
    422: {"model": ErrorResponseDetail, "description": "입력값 유효성 오류 (Unprocessable Entity)"},
    500: {"model": ErrorResponseSuccessFalse, "description": "서버 내부 오류 (Internal Server Error)"},
}

