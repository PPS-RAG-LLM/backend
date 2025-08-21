from fastapi import Request
from utils import logger
import time
from starlette.middleware.base import BaseHTTPMiddleware

logger = logger(__name__)

SLOW_MS = 1000  # 1초 이상만 INFO

class ProcessTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info("\n")
        logger.info("------------ request path: {}".format(request.url.path))
        
        start_time = time.time()
        response = await call_next(request)

        # 배포시 사용하는 로그
        # ms = (time.time() - start_time) * 1000  # ms 단위로 변환
        # response.headers["X-process-Time"] = f"{ms:.1f}"

        # if ms >= SLOW_MS:
        #     logger.info(f"slow request: {request.method} {request.url.path} {ms:.1f}ms")
        # else:
        #     logger.debug(f"process time: {ms:.1f}ms")
        
        # 개발중에만 사용하는 로그
                # 개발중에만 사용하는 로그
        process_time = time.time() - start_time
        process_time_ms = process_time * 1000
        response.headers["X-process-Time"] = f"{process_time_ms:.1f}ms"
        logger.info("------------ process time: {:.1f}ms".format(process_time_ms))
        logger.info("\n")

        return response

