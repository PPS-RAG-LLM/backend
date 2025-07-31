from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from routers.workspace import router as workspace
# from src.routes.admin import router as admin_router
# from src.routes.document import router as document_router
from errors import (
    BaseAPIException,
    base_api_exception_handler,
    general_exception_handler,
    validation_exception_handler,
)
from config import config
app = FastAPI()

app.add_exception_handler(BaseAPIException, base_api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.include_router(workspace)
# app.include_router(admin_router)
# app.include_router(document_router)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config["server"]["host"], port=config["server"]["port"])