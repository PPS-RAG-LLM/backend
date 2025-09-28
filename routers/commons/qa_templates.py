from fastapi import APIRouter, HTTPException, Path, Body
from pydantic import BaseModel
from typing import List, Optional
from service.commons.qa_templates import (
    list_qa_templates_all, generate_new_qa_prompt, update_qa_template, delete_qa_template
)
router = APIRouter(tags=["QA"], prefix="/v1/qa")

class TemplateListItem(BaseModel):
    id: int
    name: str = "qa_prompt"
    system_prompt: str = "시스템 프롬프트"
    user_prompt: Optional[str] = "유저 프롬프트"

class TemplateContentResponse(BaseModel):
    id: int
    name: str = "qa_prompt"
    system_prompt: str = "시스템 프롬프트"
    user_prompt: Optional[str] = "유저 프롬프트"

class TemplateListResponse(BaseModel):
    templates: List[TemplateContentResponse]

@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | QA 템플릿 전체 목록(상세+변수+score 포함)")
def list_qa_templates_all_route():
    items = list_qa_templates_all()
    return {"templates": items}


class CreateTemplateRequest(BaseModel):
    system_prompt: str = "시스템 프롬프트"
    user_prompt: Optional[str] = "유저 프롬프트"

@router.post("/template", response_model=TemplateContentResponse, summary="관리자용 | QA 시스템 프롬프트 생성")
def create_qa_system_prompt(body:CreateTemplateRequest):
    item = generate_new_qa_prompt(body.system_prompt, body.user_prompt)
    return item


@router.put("/template/{template_id}", response_model=TemplateContentResponse, summary="관리자용 | QA 템플릿 수정")
def update_qa_template_route(
    template_id: int = Path(..., description="프롬프트 템플릿 id"), 
    body: CreateTemplateRequest =Body(...,description="")
    ):
    item = update_qa_template(template_id, body.system_prompt, body.user_prompt)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item

@router.delete("/template/{template_id}", status_code=204, summary="관리자용 | QA 템플릿 삭제")
def delete_qa_template_route(template_id: int = Path(..., description="프롬프트 템플릿 id")):
    deleted = delete_qa_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")
