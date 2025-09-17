from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Path, Query, Body, HTTPException
from starlette.responses import StreamingResponse
from service.users.doc_generation import (
    generate_document_for_workspace,
    preflight_document_generation,
    get_document_templates,
    save_generated_document
)
from utils import logger
from errors import BadRequestError, NotFoundError
import time

logger = logger(__name__)
doc_gen_router = APIRouter(tags=["Document Generation"], prefix="/v1/workspace")

# ============================
# Request/Response Models
# ============================

class DocumentTemplate(BaseModel):
    id: int
    name: str
    category: str
    content: str
    required_vars: Optional[List[str]] = None
    is_default: bool = False

class DocumentGenerationRequest(BaseModel):
    template_id: Optional[int] = None
    template_name: Optional[str] = None
    variables: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_type: Optional[str] = None  # report, proposal, summary 등
    additional_context: Optional[str] = None
    session_id: Optional[str] = None
    reset: Optional[bool] = False

class DocumentGenerationResponse(BaseModel):
    document_id: str
    title: str
    content: str
    template_used: str
    generated_at: str
    metadata: Optional[Dict[str, Any]] = None

class DocumentTemplateResponse(BaseModel):
    templates: List[DocumentTemplate]

# ============================
# Document Generation Endpoints
# ============================

@doc_gen_router.get("/{slug}/doc-gen/templates", 
                   response_model=DocumentTemplateResponse,
                   summary="문서생성 템플릿 목록 조회")
def get_document_templates_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    document_type: Optional[str] = Query(None, description="문서 유형 필터 (report, proposal, summary 등)")
):
    """워크스페이스에서 사용 가능한 문서생성 템플릿 목록을 조회합니다."""
    try:
        user_id = 3  # TODO: 실제 인증에서 가져오기
        templates = get_document_templates(user_id, slug, document_type)
        return DocumentTemplateResponse(templates=templates)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get document templates: {e}")
        raise HTTPException(status_code=500, detail="템플릿 조회에 실패했습니다.")

@doc_gen_router.post("/{slug}/doc-gen/generate",
                    summary="문서 생성 (스트리밍)")
def generate_document_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: DocumentGenerationRequest = Body(..., description="문서 생성 요청")
):
    """워크스페이스에서 문서를 생성합니다. 스트리밍 방식으로 응답합니다."""
    try:
        user_id = 3  # TODO: 실제 인증에서 가져오기
        
        # 사전 검증
        preflight_document_generation(
            user_id=user_id,
            slug=slug,
            body=body.model_dump(exclude_unset=True)
        )
        
        # 문서 생성 스트림
        gen = generate_document_for_workspace(
            user_id=user_id,
            slug=slug,
            body=body.model_dump(exclude_unset=True)
        )
        
        return StreamingResponse(
            _stream_document_generation(gen), 
            media_type="text/event-stream"
        )
        
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        raise HTTPException(status_code=500, detail="문서 생성에 실패했습니다.")

@doc_gen_router.post("/{slug}/doc-gen/save",
                    response_model=DocumentGenerationResponse,
                    summary="생성된 문서 저장")
def save_document_endpoint(
    slug: str = Path(..., description="워크스페이스 슬러그"),
    body: DocumentGenerationResponse = Body(..., description="저장할 문서 데이터")
):
    """생성된 문서를 데이터베이스에 저장합니다."""
    try:
        user_id = 3  # TODO: 실제 인증에서 가져오기
        
        result = save_generated_document(
            user_id=user_id,
            slug=slug,
            document_data=body.model_dump(exclude_unset=True)
        )
        
        return result
        
    except BadRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Document save failed: {e}")
        raise HTTPException(status_code=500, detail="문서 저장에 실패했습니다.")

# ============================
# Helper Functions
# ============================

def _stream_document_generation(gen):
    """문서 생성 스트림을 처리합니다."""
    buf = []
    last_flush = time.monotonic()
    
    for chunk in gen:
        if not chunk:
            continue
            
        logger.debug(f"[doc_gen_chunk] {repr(chunk)}")
        
        if not buf:
            chunk = chunk.lstrip()
        buf.append(chunk)
        text = "".join(buf)
        
        # 적절한 시점에 플러시
        if (len(text) >= 32 or 
            text.endswith((" ", "\n", ".", "?", "!", "…", "。", "！", "？")) or 
            time.monotonic() - last_flush > 0.2):
            
            yield f"data: {text}\n\n"
            buf.clear()
            last_flush = time.monotonic()
    
    # 마지막 버퍼 처리
    if buf:
        text = "".join(buf)
        logger.info(f"[doc_gen_flush_end] {repr(text)}")
        yield f"data: {text}\n\n"
