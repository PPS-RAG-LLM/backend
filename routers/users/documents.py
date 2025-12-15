from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from errors.exceptions import NotFoundError
from repository.documents import get_documents_by_ids
from storage.db_models import DocumentType
from utils import logger
from pydantic import BaseModel
from typing import List
from service.manage_documents import upload_documents, delete_documents_by_ids
from utils.auth import get_user_id_from_cookie
from utils.documents import save_raw_file
from config import config

logger = logger(__name__)
router = APIRouter(tags=["Document"], prefix="/v1/document")

@router.post("/upload", summary="새 파일 업로드 및 (옵션) 워크스페이스 임베딩 준비")
async def upload_endpoint(
    files: List[UploadFile] = File(...),
    addToWorkspaces: Optional[str] = Form(None),
    user_id: int = Depends(get_user_id_from_cookie),
):
    # TODO: 인증 연동 시 user_id 추출로 교체
    logger.info(
        f"upload_endpoint: user_id={user_id}, files={len(files)}, addToWorkspaces={addToWorkspaces}"
    )
    USER_RAW_DATA_DIR = Path(config.get("user_raw_data_dir", "storage/raw_files/user_raw_data"))
    
    rel_paths = [] # RAW 저장
    for f in files:
        data = await f.read()
        saved_name = save_raw_file(f.filename, folder=USER_RAW_DATA_DIR, content=data)
        # 서비스 함수는 전체 경로(또는 상대경로)를 그대로 target_path로 사용하므로,
        # 저장된 디렉토리(USER_RAW_DATA_DIR)를 포함한 경로를 넘겨야 루트 저장을 방지함.
        full_path = USER_RAW_DATA_DIR / saved_name
        rel_paths.append(str(full_path))
        await f.seek(0)
        
    return await upload_documents(
        user_id=user_id,
        files=files,
        raw_paths=rel_paths,
        add_to_workspaces=addToWorkspaces,
    )

class TempCleanupBody(BaseModel):
    workspaceSlug: str = None
    docIds: Optional[List[str]]
    model_config = {
        "json_schema_extra": {
            "example": {
                "workspaceSlug": "my-workspace-slug",
                "docIds": ["doc_12345", "doc_67890"]
            }
        }
    }

@router.delete("/delete-documents", summary="서버문서 삭제 / 워크스페이스 문서+임베딩 삭제 || 임시 문서 삭제")
async def temp_cleanup_endpoint(body: TempCleanupBody, user_id: int = Depends(get_user_id_from_cookie)):
    logger.info(f"temp_cleanup_endpoint: body={body}")
    return delete_documents_by_ids(
        doc_ids=body.docIds, workspace_slug=body.workspaceSlug, user_id=user_id,
        )



@router.get("/download", summary="파일 다운로드 (모든 경로 지원)")
def download_endpoint(
    docId: Optional[str]    = Query(None, description="문서 ID"),
    filename: Optional[str] = Query(None, description="파일 이름"),
    user_id: int = Depends(get_user_id_from_cookie),
):
    """
    docId를 통해 파일의 정확한 위치(User/Admin/Test)를 찾아 반환합니다.
    docId가 없으면 filename으로 모든 경로를 검색합니다.
    """
    
    # 1. config에서 3가지 경로 로드
    dirs = {
        "user": Path(config.get("user_raw_data_dir")),
        "admin": Path(config.get("admin_raw_data_dir")),
        "test": Path(config.get("test_llm_raw_data_dir")),
    }
    target_file = None
    
    # 2. docId가 있는 경우: DB에서 경로 정보 조회 (가장 정확함)
    if docId:
        doc_info = get_documents_by_ids([docId])
        if doc_info and docId in doc_info:
            doc = doc_info[docId]
            doc_type = doc.get("doc_type")
            # DB에 저장된 실제 파일명 (혹은 source_path)
            # source_path가 있으면 그것을 우선, 없으면 filename 사용
            real_filename = doc.get("payload", {}).get("doc_info_path") or doc.get("filename") or filename
            
            # 타입에 따른 디렉토리 선택
            base_dir = dirs["user"] # 기본값
            if doc_type == DocumentType.ADMIN.value:
                base_dir = dirs["admin"]
            elif doc_type == DocumentType.LLM_TEST.value:
                base_dir = dirs["test"]
            
            # 파일 경로 조합
            if real_filename:
                candidate = base_dir / Path(real_filename).name # 경로 조작 방지 위해 .name 사용
                if candidate.exists():
                    target_file = candidate

    # 3. docId 조회 실패 또는 docId 미제공 시: filename으로 모든 폴더 순차 검색 (Fallback)
    if not target_file and filename:
        # User -> Admin -> Test 순서로 검색
        for key in ["user", "admin", "test"]:
            candidate = dirs[key] / filename
            if candidate.exists():
                target_file = candidate
                break

    # 4. 결과 반환
    if target_file and target_file.is_file():
        return FileResponse(
            path=target_file,
            filename=filename or target_file.name, # 다운로드될 파일명
            media_type="application/octet-stream"
        )
    
    raise NotFoundError("파일을 찾을 수 없습니다.")