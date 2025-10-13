from fastapi import Request
from utils import logger
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send, Message
import json

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
        query_string = scope.get("query_string", b"").decode()
        
        logger.info("================================")
        logger.info(f"REQUEST: {method} {path}")
        if query_string:
            logger.debug(f"QUERY: {query_string}")
        
        # Request body 읽기
        body = b""
        
        async def receive_wrapper():
            nonlocal body
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
            return message
        
        start_time = time.time()
        response_body = b""
        status_code = None
        
        # Response 가로채기
        async def send_wrapper(message: Message) -> None:
            nonlocal response_body, status_code
            
            if message["type"] == "http.response.start":
                status_code = message.get("status")
                
                # Process time 계산
                ms = (time.time() - start_time) * 1000
                
                # 헤더 추가
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", f"{ms:.1f}ms".encode()))
                message["headers"] = headers
                
                # 로깅
                if ms >= SLOW_MS:
                    logger.info(f"SLOW REQUEST: {method} {path} {ms:.1f}ms")
                
                logger.info(f"PROCESS TIME: {ms:.1f}ms")
                
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
                
                # 응답이 완료되면 로깅
                if not message.get("more_body", False):
                    # Content-Type 체크 (바이너리 요청 필터링)
                    headers = dict(scope.get("headers", []))
                    content_type = headers.get(b"content-type", b"").decode().lower()
                    
                    is_binary = any([
                        "multipart/form-data" in content_type,
                        "application/octet-stream" in content_type,
                        "/upload" in path,  # 업로드 경로는 건너뛰기
                    ])
                    
                    # Request body 로깅
                    if body and not is_binary:
                        try:
                            body_json = json.loads(body.decode())
                            logger.debug(f"REQUEST BODY: {json.dumps(body_json, ensure_ascii=False, indent=2)}")
                        except UnicodeDecodeError:
                            logger.debug(f"REQUEST BODY: <binary data, {len(body)} bytes>")
                        except:
                            logger.debug(f"REQUEST BODY: {body.decode()[:500]}")  # 최대 500자
                    elif body and is_binary:
                        logger.debug(f"REQUEST BODY: <file upload, {len(body)} bytes>")
                    
                    # Response 로깅
                    logger.info(f"RESPONSE STATUS: {status_code}")
                    if response_body:
                        try:
                            resp_json = json.loads(response_body.decode())
                            logger.debug(f"RESPONSE BODY: {json.dumps(resp_json, ensure_ascii=False, indent=2)}")
                        except:
                            logger.debug(f"RESPONSE BODY: <binary or large response, {len(response_body)} bytes>")
                    
                    logger.info("================================")
            
            await send(message)
        await self.app(scope, receive_wrapper, send_wrapper)