from fastapi import APIRouter

router = APIRouter(tags=["administrator"], prefix="/v1/admin")

@router.get("/")
def admin_endpoint():
    return {"message": "Admin page"}