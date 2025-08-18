# routers/admin/manage_admin_LLM_api.py
from fastapi import APIRouter, Query
from pydantic import BaseModel

from service.admin.manage_admin_LLM import (
    TopKSettingsBody,
    ModelLoadBody,
    CreatePromptBody,
    UpdatePromptBody,
    CompareModelsBody,
    set_topk_settings,
    get_model_list,
    load_or_unload_model,
    compare_models,
    list_prompts,
    create_prompt,
    get_prompt,
    update_prompt,
    delete_prompt,
    test_prompt,
)

router = APIRouter(prefix="/v1/admin/llm", tags=["Admin LLM"], responses={200: {"description": "Success"}})

@router.post("/settings")
def set_settings(body: TopKSettingsBody):
    return set_topk_settings(body.topK)

@router.get("/settings/model-list")
def model_list(category: str = Query(..., description="qa | doc_gen | summary")):
    return get_model_list(category)

@router.put("/settings/model-load")
def model_load(category: str = Query(..., description="qa | doc_gen | summary"), body: ModelLoadBody = ...):
    return load_or_unload_model(category, body.modelName)

@router.get("/compare-models")
def compare_models_list(category: str = Query(...)):
    # 기존 사양 유지: querystring만 받음
    return compare_models(CompareModelsBody(category=category))

@router.get("/prompts")
def get_prompts(category: str = Query(...)):
    return list_prompts(category)

@router.post("/prompts")
def create_prompt_route(category: str = Query(...), body: CreatePromptBody = ...):
    return create_prompt(category, body)

@router.get("/prompt/{prompt_id}")
def get_prompt_route(prompt_id: int):
    return get_prompt(prompt_id)

@router.put("/prompt/{prompt_id}")
def update_prompt_route(prompt_id: int, body: UpdatePromptBody = ...):
    return update_prompt(prompt_id, body)

@router.delete("/prompt/{prompt_id}")
def delete_prompt_route(prompt_id: int):
    return delete_prompt(prompt_id)

@router.post("/prompt/{prompt_id}")
def test_prompt_route(prompt_id: int, body: dict | None = None):
    return test_prompt(prompt_id, body)
