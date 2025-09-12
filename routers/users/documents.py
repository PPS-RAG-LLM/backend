from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from service.users.documents import upload_documents
from utils import logger
from pydantic import BaseModel
from typing import List
from service.users.documents import delete_documents_by_ids

logger = logger(__name__)
router = APIRouter(tags=["document"], prefix="/v1/document")

@router.post("/upload", summary="새 파일 업로드 및 (옵션) 워크스페이스 임베딩 준비")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    addToWorkspaces: Optional[str] = Form(None),
):
    # TODO: 인증 연동 시 user_id 추출로 교체
    user_id = 3
    logger.info(
        f"upload_endpoint: user_id={user_id}, files={len(files)}, addToWorkspaces={addToWorkspaces}"
    )
    return await upload_documents(
        user_id=user_id,
        files=files,
        add_to_workspaces=addToWorkspaces,
    )

class TempCleanupBody(BaseModel):
    workspaceSlug: str = None
    docIds: Optional[List[str]]

@router.delete("/delete-documents", summary="서버문서 삭제 / 워크스페이스 문서+임베딩 삭제 || 임시 문서 삭제")
async def temp_cleanup_endpoint(body: TempCleanupBody):
    user_id = 3
    logger.info(f"temp_cleanup_endpoint: body={body}")
    return delete_documents_by_ids(
        doc_ids=body.docIds, workspace_slug=body.workspaceSlug, user_id=user_id
        )
