from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from service.users.summary_templates import (
    list_summary_templates,
    get_summary_template,
)

router = APIRouter(tags=["summary"], prefix="/v1/summary")

class TemplateListItem(BaseModel):
    id: int
    name: str
    content: str

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class TemplateContentResponse(BaseModel):
    id: int
    name: str
    content: str

@router.get("/templates", response_model=TemplateListResponse, summary="요약용 템플릿 전체 목록(상세 포함)")
def list_summary_templates_route():
    items = list_summary_templates()
    return {"templates": items}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="선택 템플릿 콘텐츠 조회")
def get_summary_template_route(template_id: int):
    row = get_summary_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"id": row["id"], "name": row["name"], "content": row["content"]}