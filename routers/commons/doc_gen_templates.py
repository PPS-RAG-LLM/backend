# /home/work/CoreIQ/yb/backend/routers/users/doc_gen_templates.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.commons.doc_gen_templates import (
    list_doc_gen_templates,
    get_doc_gen_template,
    list_doc_gen_templates_all,
    generate_new_doc_gen_prompt
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
    user_prompt: Optional[str] = ""
    variables: List[VariableItem]

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class TemplateContentResponse(BaseModel):
    id: int
    name: str
    system_prompt: str
    user_prompt: Optional[str] = ""
    variables: List[VariableItem]

@router.get("/templates/is_default", response_model=TemplateListResponse, summary="사용자용 | 관리자 측에서 기본값으로 정해진 문서생성용 템플릿 각 세부 테스크 목록")
def list_doc_gen_templates_route():
    items = list_doc_gen_templates()
    return {"templates": items}

@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | 문서생성용 템플릿 전체 목록(상세+변수+score 포함)")
def list_doc_gen_templates_all_route():
    items = list_doc_gen_templates_all()
    return {"templates": items}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="문서생성 템플릿 단건(상세+변수)")
def get_doc_gen_template_route(template_id: int):
    row = get_doc_gen_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return row

class TemplateListItem(BaseModel):
    name: str 
    system_prompt: str
    user_prompt: Optional[str] = ""
    variables: List[VariableItem]

@router.post("/template")
def create_doc_gen_prompt(body:TemplateListItem):
    name = body.name
    system_prompt = body.system_prompt
    user_prompt = body.user_prompt
    variables = [var.model_dump() for var in body.variables]
    item = generate_new_doc_gen_prompt(name, system_prompt, user_prompt, variables)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"templates": item}