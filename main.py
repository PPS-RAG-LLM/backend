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
from routers.users.workspace_thread import thread_router
from routers.users.documents import router as document_router
from routers.users.chat_feedback import feedback_router
from routers.admin.LLM_finetuning_api import router as llm_finetuning_router
from routers.admin.manage_admin_LLM_api import router as admin_llm_router
from routers.admin.manage_test_LLM_api import router as test_llm_router
# from src.routes.admin import router as admin_router
# from src.routes.document import router as document_router
from routers.login.sso import sso_router as sso_router
from routers.login.mock_company import mock_company_router as mock_company_router
from utils import logger, init_db
from contextlib import asynccontextmanager
import asyncio

from routers.commons.summary_templates import router as summary_router
from routers.commons.doc_gen_templates import router as doc_gen_templates_router
from routers.commons.qa_templates import router as qa_templates_router
from routers.test_error.test_error import test_error_router as test_error_router
logger = logger(__name__)

# === [ADD] 사용자 관리 라우터 임포트 ===
from routers.admin.manage_user_api import router as admin_user_router
# =====================================


######################### Session Cleaner #########################

@asynccontextmanager
async def lifspan(app):
    """주기적으로 만료된 세션 정리"""
    init_db() # 스키마 1회 초기화, 이미 있으면 즉시 스킵
    
    # # 서버 시작 시 활성 임베딩 모델 확인 및 rag_settings 동기화
    # try:
    #     from utils.database import get_session
    #     from storage.db_models import EmbeddingModel, RagSettings
    #     from datetime import datetime
    #     with get_session() as session:
    #         row = (
    #             session.query(EmbeddingModel)
    #             .filter(EmbeddingModel.is_active == 1)
    #             .order_by(EmbeddingModel.activated_at.desc().nullslast())
    #             .first()
    #         )
    #         if row:
    #             active_key = row.name
    #             s = session.query(RagSettings).filter(RagSettings.id == 1).first()
    #             if not s:
    #                 s = RagSettings(id=1)
    #                 session.add(s)
    #             s.embedding_key = active_key
    #             s.updated_at = datetime.utcnow()
    #             session.commit()
    #             logger.info(f"활성 임베딩 모델 확인: {active_key}")
    #         else:
    #             logger.warning("활성 임베딩 모델이 없습니다. embedding_models에서 is_active=1을 하나 지정하세요.")
    # except Exception as e:
    #     logger.error(f"임베딩 모델 확인 실패: {e}")

    # (선택) 프리로드는 지연 로딩으로 대체 가능
    try:
        from service.admin.manage_vator_DB import warmup_active_embedder
        logger.info("활성 임베딩 모델 확인 및 (선택) 프리로드...")
        # warmup_active_embedder(logger.info)
    except Exception as e:
        logger.error(f"임베딩 모델 확인/프리로드 실패: {e}")

    # ====== LLM 활성 목록 로깅(로드는 하지 않음) ======
    try:
        from service.admin.manage_admin_LLM import get_active_llm_models
        active_llms = get_active_llm_models()
        if active_llms:
            logger.info("활성 LLM 모델(로드 지연, 최초 사용 시 로드):")
            for m in active_llms:
                logger.info(
                    f" - id={m.get('id')} name={m.get('name')} category={m.get('category')} path={m.get('model_path')}"
                )
        else:
            logger.warning("활성 LLM 모델이 없습니다. llm_models.is_active=1을 설정하세요.")
    except Exception as e:
        logger.error(f"LLM 활성 목록 로깅 실패: {e}")
    
    from repository.session import cleanup_expired_sessions
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
# import sys
# if sys.version_info >= (3, 11):
#     app.add_exception_handler(BaseExceptionGroup, general_exception_handler)

app.add_exception_handler(BaseAPIException, base_api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(NotFoundError, not_found_error_handler)


######################### Router ######################### 

# 라우터 등록
app.include_router(workspace)
app.include_router(document_router)
app.include_router(workspace_singular)
app.include_router(chat_router)
app.include_router(thread_router)
app.include_router(vector_db_router)
app.include_router(feedback_router)
app.include_router(llm_finetuning_router)
app.include_router(admin_llm_router)
app.include_router(test_llm_router)
app.include_router(sso_router)
app.include_router(mock_company_router)
app.include_router(summary_router)
app.include_router(doc_gen_templates_router)
app.include_router(qa_templates_router)
# app.include_router(admin_router)
# app.include_router(document_router)
# === [ADD] 사용자 관리 라우터 등록 ===
app.include_router(admin_user_router)
app.include_router(test_error_router)
# ===================================


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
        reload_excludes=[
            "**/unsloth_compiled_cache/**",
            "**/storage/model/**",
            "**/storage/train_data/**",
        ],
        reload_dirs=["routers", "repository", "service", "utils"],
    )