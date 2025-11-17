"""
에러 관리 모듈
쉬운 임포트와 사용을 위한 패키지 초기화
"""

from .exceptions import (
    BaseAPIException,
    BadRequestError,
    UnauthorizedError, 
    ForbiddenError,
    NotFoundError,
    UnprocessableEntityError,
    InternalServerError,
    DatabaseError,
    SessionNotFoundError,
    NotImplementedAPIError,
    DocumentProcessingError
)

from .handlers import (
    base_api_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    not_found_error_handler,
)

# 자주 사용되는 예외들을 쉽게 접근할 수 있도록 별칭 제공
BadRequest = BadRequestError
Unauthorized = UnauthorizedError
Forbidden = ForbiddenError
NotFound = NotFoundError
InternalServer = InternalServerError
SessionNotFound = SessionNotFoundError
NotImplementedAPI = NotImplementedAPIError

__all__ = [
    # 예외 클래스들
    "BaseAPIException",
    "BadRequestError", "BadRequest",
    "ForbiddenError", "Forbidden",
    "NotFoundError", "NotFound",
    "InternalServerError", "InternalServer",
    "DatabaseError",
    "UnauthorizedError", "Unauthorized", 
    "SessionNotFoundError", "SessionNotFound",
    "NotImplementedAPIError", "NotImplementedAPI",
    "DocumentProcessingError", "DocumentProcessingError",
    # 핸들러들
    "base_api_exception_handler",
    "general_exception_handler", 
    "validation_exception_handler",
    "not_found_error_handler",
]