from fastapi import Request
from utils import logger
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send, Message
logger = logger(__name__)

SLOW_MS = 1000  # 1초 이상만 INFO


class ProcessTimeMiddleware:
    """순수 ASGI middleware - Python 3.13 호환"""
    
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Request 정보 추출
        path = scope.get("path", "")
        method = scope.get("method", "")
        logger.info("================================")
        logger.info(f"REQUEST PATH: {method} {path} ")
        
        start_time = time.time()
        
        # Response 가로채기
        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                # Process time 계산
                ms = (time.time() - start_time) * 1000
                
                # 헤더 추가
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{ms:.1f}ms".encode()))
                message["headers"] = headers
                
                # 로깅
                if ms >= SLOW_MS:
                    logger.info(f"SLOW REQUEST: {method} {path} {ms:.1f}ms")
                # else:
                #     logger.debug(f"PROCESS TIME: {ms:.1f}ms")
                
                logger.info(f"PROCESS TIME: {ms:.1f}ms")
                logger.info("================================")
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
