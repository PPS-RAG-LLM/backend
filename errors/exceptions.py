"""
전역 에러 관리를 위한 커스텀 예외 클래스들
API 명세서에 맞는 표준 에러 형식을 제공
"""

class BaseAPIException(Exception):
    """기본 예외 클래스"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class BadRequestError(BaseAPIException):
    """400 Bad Request - 잘못된 요청"""
    def __init__(self, message: str = "잘못된 요청입니다"):
        super().__init__(message, 400)


class UnauthorizedError(BaseAPIException):
    """401 Unauthorized - 인증되지 않은 요청"""
    def __init__(self, message: str = "인증이 필요합니다"):
        super().__init__(message, 401)

class SessionNotFoundError(BaseAPIException):
    """세션을 찾을 수 없음"""
    def __init__(self, message: str):
        super().__init__(message, 401)
    
class InvalidSessionError(BaseAPIException):
    """잘못된 세션"""

class ForbiddenError(BaseAPIException):
    """403 Forbidden - 권한 없음"""
    def __init__(self, message: str = "권한이 없습니다"):
        super().__init__(message, 403)

class NotFoundError(BaseAPIException):
    """404 Not Found - 리소스를 찾을 수 없음"""
    def __init__(self, message: str = "요청한 리소스를 찾을 수 없습니다"):
        super().__init__(message, 404)


class UnprocessableEntityError(BaseAPIException):
    """422 Unprocessable Entity - 처리할 수 없는 엔티티"""
    def __init__(self, message: str = "요청을 처리할 수 없습니다"):
        super().__init__(message, 422)


class InternalServerError(BaseAPIException):
    """500 Internal Server Error - 서버 내부 오류"""
    def __init__(self, message: str = "서버 내부 오류가 발생했습니다"):
        super().__init__(message, 500)


# 특정 도메인 예외들
class WorkspaceNotFoundError(NotFoundError):
    """워크스페이스를 찾을 수 없음"""
    def __init__(self, slug: str):
        super().__init__(f"워크스페이스를 찾을 수 없습니다: {slug}")


class WorkspaceAlreadyExistsError(BadRequestError):
    """워크스페이스가 이미 존재함"""
    def __init__(self, slug: str):
        super().__init__(f"이미 사용중인 slug입니다: {slug}")


class InvalidAPIKeyError(ForbiddenError):
    """잘못된 API 키"""
    def __init__(self):
        super().__init__("Invalid API Key")


class ModelNotSupportedError(BadRequestError):
    """지원하지 않는 모델"""
    def __init__(self, model_name: str):
        super().__init__(f"지원하지 않는 모델입니다: {model_name}")


class DatabaseError(InternalServerError):
    """데이터베이스 오류"""
    def __init__(self, message: str = "데이터베이스 처리 중 오류가 발생했습니다"):
        super().__init__(f"데이터베이스 오류: {message}")


