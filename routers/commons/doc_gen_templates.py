# /home/work/CoreIQ/yb/backend/routers/users/doc_gen_templates.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.users.doc_gen_templates import (
    list_doc_gen_templates,
    get_doc_gen_template,
    list_doc_gen_templates_all,
)

router = APIRouter(tags=["doc_gen"], prefix="/v1/doc-gen")

class VariableItem(BaseModel):
    type: str
    key: str
    value: Optional[str] = None
    description: str

class TemplateListItem(BaseModel):
    id: int
    name: str
    system_prompt: str
    user_prompt: str | None = None
    variables: List[VariableItem]

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class TemplateContentResponse(BaseModel):
    id: int
    name: str
    system_prompt: str
    user_prompt: str | None = None
    variables: List[VariableItem]

@router.get("/templates", response_model=TemplateListResponse, summary="문서생성용 템플릿 각 세부 테스크 only default 목록(상세+변수 포함)")
def list_doc_gen_templates_route():
    items = list_doc_gen_templates()
    return {"templates": items}

@router.get("/templates/all", response_model=TemplateListResponse, summary="문서생성용 템플릿 전체 목록(상세+변수 포함)")
def list_doc_gen_templates_all_route():
    items = list_doc_gen_templates_all()
    return {"templates": items}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="문서생성 템플릿 단건(상세+변수)")
def get_doc_gen_template_route(template_id: int):
    row = get_doc_gen_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return row