from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from service.commons.qa_templates import list_qa_templates_all

router = APIRouter(tags=["QA"], prefix="/v1/qa")

class TemplateListItem(BaseModel):
    id: int
    name: str
    system_prompt: str

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]



@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | QA 템플릿 전체 목록(상세+변수+score 포함)")
def list_qa_templates_all_route():
    items = list_qa_templates_all()
    return {"templates": items}