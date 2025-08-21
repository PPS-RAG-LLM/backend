from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from config import config
from errors import (
    BaseAPIException,
    base_api_exception_handler,
    general_exception_handler,
    validation_exception_handler,
    NotFoundError,
    not_found_error_handler,
)
from utils import ProcessTimeMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from routers.users.workspace import router as workspace, router_singular as workspace_singular
from routers.users.workspace_chat import chat_router as chat_router
from routers.admin.manage_vator_DB_api import router as vector_db_router
# from src.routes.admin import router as admin_router
# from src.routes.document import router as document_router
from routers.sso import sso_router as sso_router
from routers.mock_company import mock_company_router as mock_company_router
from utils import logger, init_db
from contextlib import asynccontextmanager
import asyncio
logger = logger(__name__)


######################### Session Cleaner #########################

@asynccontextmanager
async def lifspan(app):
    """주기적으로 만료된 세션 정리"""
    init_db() # 스키마 1회 초기화, 이미 있으면 즉시 스킵
    from repository.users.session import cleanup_expired_sessions
    async def _periodic_db_session_cleanup():
        while True:
            try:
                cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"session cleaner error: {e}")
            await asyncio.sleep(3000) # 3000초마다 정리
    logger.info("lifspan start")
    app.state.session_cleanup_task = asyncio.create_task(_periodic_db_session_cleanup())
    yield
    app.state.session_cleanup_task.cancel()
    try:
        await app.state.session_cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifspan)

######################### Middleware ######################### 

# 미들웨어 설정
server_conf = config.get("server")
cors_origins = server_conf.get("cors_origins")
trusted_hosts = server_conf.get("trusted_hosts")
force_https = server_conf.get("force_https")
gzip_min_size = server_conf.get("gzip_min_size")

# 미들웨어 등록
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )
app.add_middleware(GZipMiddleware, minimum_size=gzip_min_size)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)
if force_https:
    app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(ProcessTimeMiddleware)

######################### Exception Handler ######################### 

app.add_exception_handler(BaseAPIException, base_api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(NotFoundError, not_found_error_handler)


######################### Router ######################### 

# 라우터 등록
app.include_router(workspace)
app.include_router(workspace_singular)
app.include_router(chat_router)
app.include_router(vector_db_router)
app.include_router(sso_router)
app.include_router(mock_company_router)
# app.include_router(admin_router)
# app.include_router(document_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host=config["server"]["host"], 
        port=config["server"]["port"],
        reload=True,
    )