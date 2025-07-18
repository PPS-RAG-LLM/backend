from fastapi import APIRouter

router = APIRouter(tags=["documents"], prefix="/docs")

@router.get("/")
def document_endpoint():
    return {"message": "Document page"}