from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from service.users.upload_documents import upload_document
from utils import logger

logger = logger(__name__)

router = APIRouter(tags=["document"], prefix="/v1/document")


@router.post("/upload", summary="새 파일 업로드 및 (옵션) 워크스페이스 임베딩 준비")
async def upload_endpoint(
    file: UploadFile = File(...),
    addToWorkspaces: Optional[str] = Form(None),
):
    # TODO: 인증 연동 시 user_id 추출로 교체
    user_id = 3
    logger.info(f"upload_endpoint: user_id={user_id}, file={file}, addToWorkspaces={addToWorkspaces}")
    return await upload_document(
        user_id=user_id,
        file=file,
        add_to_workspaces=addToWorkspaces,
    )