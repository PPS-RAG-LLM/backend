from fastapi import APIRouter

router = APIRouter(tags=["documents"], prefix="/v1/document")

@router.get("/")
def document_endpoint():
    return {"message": "Document page"}