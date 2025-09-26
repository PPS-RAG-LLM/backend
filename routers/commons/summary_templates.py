from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional
from service.commons.summary_templates import (
    generate_summary_template,
    list_summary_templates,
    list_summary_templates_all,
    get_summary_template,
    update_summary_template,
    delete_summary_template
)

router = APIRouter(tags=["summary"], prefix="/v1/summary")

class TemplateListItem(BaseModel):
    id: int
    name: str = "이름"
    system_prompt: str = "시스템 프롬프트"
    user_prompt: str ="유저 프롬프트"

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class TemplateContentResponse(BaseModel):
    id: int
    name: str = "프롬프트 name"
    system_prompt: str = "시스템 프롬프트"
    user_prompt: str = "유저 프롬프트"

@router.get("/templates/is_default", response_model=TemplateListResponse, summary="사용자용 | 요약용 템플릿 전체 목록(상세 포함)")
def list_summary_templates_route():
    items = list_summary_templates()
    return {"templates": items}
    
@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | 요약용 템플릿 전체 목록(상세 포함)")
def list_summary_templates_route():
    items = list_summary_templates_all()
    return {"templates": items}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="선택 템플릿 콘텐츠 조회")
def get_summary_template_route(template_id: int):
    row = get_summary_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"id": row["id"], "name": row["name"], "system_prompt": row["system_prompt"], "user_prompt": row["user_prompt"]}

class CreateTemplateRequest(BaseModel):
    system_prompt: str ="시스템 프롬프트"
    user_prompt: str  ="유저 프롬프트"

@router.post("/template", summary="관리자용 | QA 시스템 프롬프트 생성")
def create_qa_system_prompt(body:CreateTemplateRequest):
    item = generate_summary_template(body.system_prompt, body.user_prompt)
    return item

@router.put("/template/{template_id}", response_model=TemplateContentResponse, summary="관리자용 | 요약 템플릿 수정")
def update_summary_template_route(template_id: int, body: CreateTemplateRequest):
    item = update_summary_template(template_id, body.system_prompt, body.user_prompt)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item

@router.delete("/template/{template_id}", status_code=204, summary="관리자용 | 요약 템플릿 삭제")
def delete_summary_template_route(template_id: int):
    deleted = delete_summary_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")
