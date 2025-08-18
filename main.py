from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from config import config
from errors import (
    BaseAPIException,
    base_api_exception_handler,
    general_exception_handler,
    validation_exception_handler,
)
from utils import ProcessTimeMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from routers.users.workspace import router as workspace, router_singular as workspace_singular
from routers.admin.manage_vator_DB_api import router as vector_db_router
# from src.routes.admin import router as admin_router
# from src.routes.document import router as document_router
from routers.sso import sso_router as sso_router
from routers.mock_company import mock_company_router as mock_company_router
app = FastAPI()

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


######################### Router ######################### 

# 라우터 등록
app.include_router(workspace)
app.include_router(workspace_singular)
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