# /home/work/CoreIQ/yb/backend/routers/users/doc_gen_templates.py
from fastapi import APIRouter, HTTPException, Body, Path, Query
from pydantic import BaseModel
from typing import List, Optional
from service.commons.doc_gen_templates import (
    list_doc_gen_templates,
    get_doc_gen_template,
    list_doc_gen_templates_all,
    generate_new_doc_gen_prompt,
    update_doc_gen_prompt_service,
    remove_doc_gen_template,
    remove_doc_gen_prompt_variable,
    create_doc_gen_prompt_variable_service
)

router = APIRouter(tags=["doc_gen"], prefix="/v1/doc-gen")

class VariableItem(BaseModel):
    id: int
    type: str = "'start_date'| 'end_date' | 'date' | 'text' | 'textarea' | 'number'"
    key: str = "'시작일' | '종료일' | '날짜' | '작성자' | '요청사항' 등 사용자에게 표시될 부분 "
    value: Optional[str] = "관리자 테스트 시 쓰일 예시 답안"
    description: str = "사용자에게 보여줄 설명"
    required: Optional[bool] = False

class TemplateListItem(BaseModel):
    id: int
    name: str = "'business_trip' | 'report' | 'meeting'"
    systemPrompt: str = "시스템 프롬프트"
    userPrompt: Optional[str] = "유저 프롬프트"
    variables: List[VariableItem]

class TemplateListResponse(BaseModel):
    templates: List[TemplateListItem]

class TemplateContentResponse(BaseModel):
    name: str = "'business_trip' | 'report' | 'meeting'"
    systemPrompt: str ="시스템 프롬프트" 
    userPrompt: Optional[str] = "유저 프롬프트"
    variables: List[VariableItem]

@router.get("/templates/is_default", response_model=TemplateListResponse, summary="사용자용 | 관리자 측에서 기본값으로 정해진 문서생성용 템플릿 각 세부 테스크 목록")
def list_doc_gen_templates_route():
    items = list_doc_gen_templates()
    return {"templates": items}

@router.get("/templates/all", response_model=TemplateListResponse, summary="관리자용 | 문서생성용 템플릿 전체 목록 출력 (상세+변수포함)")
def list_doc_gen_templates_all_route():
    items = list_doc_gen_templates_all()
    return {"templates": items}

@router.get("/template/{template_id}", response_model=TemplateContentResponse, summary="문서생성성 템플릿 단건 출력(상세+변수)")
def get_doc_gen_template_route(template_id: int = Path(..., description="프롬프트 템플릿 id")):
    row = get_doc_gen_template(template_id)
    if not row:
        raise HTTPException(status_code=404, detail="Template not found")
    return row

class VariableItemRequest(BaseModel):
    type: str = "'start_date'| 'end_date' | 'date' | 'text' | 'textarea' | 'number'"
    key: str = "'시작일' | '종료일' | '날짜' | '작성자' | '요청사항' 등 사용자에게 표시될 부분 "
    value: Optional[str] = "관리자 테스트 시 쓰일 예시 답안"
    description: str = "사용자에게 보여줄 설명"
    required: Optional[bool] = False

class TemplatePayload(BaseModel):
    name: str = "'business_trip' | 'report' | 'meeting'"
    systemPrompt: str = "시스템프롬프트"
    userPrompt: Optional[str] = "유저 프롬프트트"
    variables: List[VariableItemRequest]

@router.post("/template",  response_model=TemplateContentResponse, summary="관리자용 | 문서생성용 프롬프트 템플릿 만들기")
def create_doc_gen_prompt(body:TemplatePayload):
    variables = [var.model_dump() for var in body.variables]
    item = generate_new_doc_gen_prompt(body.name, body.systemPrompt, body.userPrompt, variables)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item

@router.put("/template/{template_id}", response_model=TemplateContentResponse, summary="관리자용 | 문서생성용 프롬프트 템플릿 수정" )
def update_doc_gen_prompt(template_id: int = Path(..., description="프롬프트 템플릿 id"),
 body: TemplateContentResponse = Body(..., description="")):
    variables = [var.model_dump() for var in body.variables]
    item = update_doc_gen_prompt_service(template_id, body.name, body.systemPrompt, body.userPrompt, variables)
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item


@router.delete("/template/{template_id}", status_code=204, summary="관리자용 | 문서생성용 프롬프트 템플릿 삭제" )
def delete_doc_gen_prompt(template_id: int = Path(..., description="프롬프트 템플릿 id")):
    deleted = remove_doc_gen_template(template_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")


@router.post("/template/{template_id}/variable", response_model=VariableItem, summary="관리자용 | 문서생성용 프롬프트 템플릿 레이블 추가")
def create_doc_gen_prompt_variable(
    template_id: int = Path(..., description="프롬프트 템플릿 id"),
    variables: VariableItemRequest = Body(..., description="레이블 추가 요청")
):
    item = create_doc_gen_prompt_variable_service(template_id, variables.model_dump())
    if not item:
        raise HTTPException(status_code=404, detail="Template not found")
    return item

@router.delete("/template/{template_id}/variable", status_code=204, summary="관리자용 | 문서생성용 프롬프트 템플릿 레이블 삭제")
def delete_doc_gen_prompt_variable(
    template_id: int = Path(..., description="프롬프트 템플릿 id"),
    variable_id: int = Query(..., description="레이블 id")
):
    deleted = remove_doc_gen_prompt_variable(template_id, variable_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Template not found")

