from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.commons.qa_templates import list_qa_templates_all, generate_new_qa_prompt

router = APIRouter(tags=["QA"], prefix="/v1/qa")

class TemplateListItem(BaseModel):
    id: int
    name: str
    system_prompt: str
    user_prompt: Optional[str] = ""

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]



@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | QA 템플릿 전체 목록(상세+변수+score 포함)")
def list_qa_templates_all_route():
    items = list_qa_templates_all()
    return {"templates": items}


class CreateTemplateRequest(BaseModel):
    system_prompt: str
    user_prompt: Optional[str] = ""

@router.post("/template", summary="관리자용 | QA 시스템 프롬프트 생성")
def create_qa_system_prompt(body:CreateTemplateRequest):
    item = generate_new_qa_prompt(body.system_prompt, body.user_prompt)
    return {"templates": item}