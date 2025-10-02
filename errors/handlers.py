from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .exceptions import (
    BaseAPIException,
    ForbiddenError,
    InternalServerError,
    NotFoundError,
)

# import traceback
from utils import logger

logger = logger(__name__)


async def base_api_exception_handler(request: Request, exc: Exception):
    """커스텀 API 예외 핸들러"""

    # BaseAPIException이 아닌 경우 일반 처리
    if not isinstance(exc, BaseAPIException):
        return await general_exception_handler(request, exc)

    # 403 Forbidden은 특별한 형식
    if isinstance(exc, ForbiddenError):
        return JSONResponse(
            status_code=exc.status_code, content={"message": exc.message}
        )

    # 500 Internal Server Error는 특별한 형식
    if isinstance(exc, InternalServerError):
        logger.error(f"Internal Server Error: {exc.message}")
        logger.error(f"Request: {request.method} {request.url}")

        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.message},
        )

    # 나머지 상태 코드들은 기본 FastAPI 형식
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


async def not_found_error_handler(request: Request, exc: NotFoundError):
    logger.info(f"Handling NotFoundError: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.message},
    )


async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    import traceback
    
    # 에러 위치만 간결하게 출력
    tb = traceback.extract_tb(exc.__traceback__)
    last_frame = tb[-1] if tb else None
    
    logger.error(f"❌ {exc.__class__.__name__}: {str(exc)}")
    if last_frame:
        logger.error(f"📍 {last_frame.filename}:{last_frame.lineno} in {last_frame.name}")
    logger.error(f"🔗 {request.method} {request.url}")
    
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "예상하지 못한 서버 오류가 발생했습니다"},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )