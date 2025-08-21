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
    WorkspaceNotFoundError,
    WorkspaceAlreadyExistsError,
    InvalidAPIKeyError,
    ModelNotSupportedError,
    DatabaseError,
    SessionNotFoundError,
    InvalidSessionError,
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
UnprocessableEntity = UnprocessableEntityError
InternalServer = InternalServerError
SessionNotFound = SessionNotFoundError
InvalidSession = InvalidSessionError

__all__ = [
    # 예외 클래스들
    "BaseAPIException",
    "BadRequestError", "BadRequest",
    "UnauthorizedError", "Unauthorized", 
    "ForbiddenError", "Forbidden",
    "NotFoundError", "NotFound",
    "UnprocessableEntityError", "UnprocessableEntity",
    "InternalServerError", "InternalServer",
    "WorkspaceNotFoundError",
    "WorkspaceAlreadyExistsError", 
    "InvalidAPIKeyError",
    "ModelNotSupportedError",
    "DatabaseError",
    "SessionNotFoundError", "SessionNotFound",
    "InvalidSessionError", "InvalidSession",
    # 핸들러들
    "base_api_exception_handler",
    "general_exception_handler", 
    "validation_exception_handler",
    "not_found_error_handler",
]