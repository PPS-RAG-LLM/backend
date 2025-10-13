from fastapi import APIRouter, Query
from errors import BadRequestError, NotFoundError, InternalServerError

test_error_router = APIRouter(tags=["test_error"], prefix="/v1/test")


@test_error_router.get("/error/value")
def test_value_error():
    """ValueError 테스트 - 일반 예외"""
    raise ValueError("테스트용 ValueError 발생!")


@test_error_router.get("/error/notfound")
def test_not_found():
    """NotFoundError 테스트 - 커스텀 예외"""
    raise NotFoundError("테스트용 리소스를 찾을 수 없습니다")


@test_error_router.get("/error/badrequest")
def test_bad_request():
    """BadRequestError 테스트 - 커스텀 예외"""
    raise BadRequestError("테스트용 잘못된 요청입니다")


@test_error_router.get("/error/internal")
def test_internal_error():
    """InternalServerError 테스트 - 커스텀 예외"""
    raise InternalServerError("테스트용 서버 내부 오류입니다")


@test_error_router.get("/error/zerodivision")
def test_zero_division():
    """ZeroDivisionError 테스트 - 일반 예외"""
    result = 10 / 0
    return {"result": result}


@test_error_router.get("/error/custom")
def test_custom_error(error_type: str = Query("value", description="value|notfound|badrequest|internal|zero")):
    """동적 에러 테스트"""
    if error_type == "value":
        raise ValueError("동적 ValueError 발생!")
    elif error_type == "notfound":
        raise NotFoundError("동적 NotFoundError 발생!")
    elif error_type == "badrequest":
        raise BadRequestError("동적 BadRequestError 발생!")
    elif error_type == "internal":
        raise InternalServerError("동적 InternalServerError 발생!")
    elif error_type == "zero":
        return {"result": 10 / 0}
    else:
        return {"message": "정상 응답"}