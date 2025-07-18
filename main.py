from fastapi import FastAPI
from src.api.routes.chatbot import router as chat_router
from src.api.routes.admin import router as admin_router
from src.api.routes.document import router as document_router

app = FastAPI()
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(document_router)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

