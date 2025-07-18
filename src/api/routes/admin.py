from fastapi import APIRouter

router = APIRouter(tags=["administrator"], prefix="/admin")

@router.get("/")
def admin_endpoint():
    return {"message": "Admin page"}