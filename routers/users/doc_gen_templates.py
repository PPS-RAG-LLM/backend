# /home/work/CoreIQ/yb/backend/routers/users/doc_gen_templates.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.users.doc_gen_templates import (
    list_doc_gen_templates,
    get_doc_gen_template,
)

router = APIRouter(tags=["doc_gen"], prefix="/v1/doc-gen")

class TemplateListItem(BaseModel):
    id: int
    name: str

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class VariableItem(BaseModel):
    type: str
    key: str
    value: Optional[str] = None
    description: str

class TemplateContentResponse(BaseModel):
    id: int
    name: str
    content: str
    variables: List[VariableItem]

@router.get("/templates", response_model=TemplateListResponse, summary="문서생성용 템플릿 목록")
def list_doc_gen_templates_route():
    items = list_doc_gen_templates()
    return {"templates": [{"id": r["id"], "name": r["name"]} for r in items]}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="선택 템플릿+변수 매핑 조회")
def get_doc_gen_template_route(template_id: int):
    row = get_doc_gen_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "id": row["id"],
        "name": row["name"],
        "content": row["content"],
        "variables": row["variables"],
    }