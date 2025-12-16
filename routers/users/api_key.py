from typing import Optional
from typing_extensions import Literal
import requests
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from utils import logger

logger = logger(__name__)

# prefix를 변경하여 충돌 방지 및 명확화 (선택 사항이나 권장됨)
api_key_router = APIRouter(tags=["Workspace"], prefix="/v1/workspace") 


class ApiKeyCheckBody(BaseModel):
    apiKey: str
    provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai", 
        description="Provider name (openai, anthropic, gemini)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "apiKey": "sk-...",
                "provider": "openai"
            }
        }
    }

class ApiKeyCheckResponse(BaseModel):
    valid: bool
    reason: Optional[str] = None
    model_config = {
        "json_schema_extra": {
            "example": {
                "valid": False,
                "reason": "INVALID_KEY"
            }
        }
    }

@api_key_router.post("/check-api-key", summary="API Key 유효성 검사", response_model=ApiKeyCheckResponse)
def check_api_key(body: ApiKeyCheckBody) -> ApiKeyCheckResponse:
    """
    API Key 유효성 검사
    POST /v1/workspace/check-api-key
    { "apiKey": "sk-...", "provider": "openai" }
    """
    api_key = body.apiKey
    provider = body.provider.lower()
    
    url = ""
    headers = {}
    params = {}

    # Provider별 설정
    if provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
    
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"  # Anthropic 필수 헤더
        }
        
    elif provider == "gemini":
        # Gemini는 일반적으로 쿼리 파라미터로 키를 전달합니다.
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        params = {"key": api_key}
        
    else:
        return {"valid": False, "reason": "UNSUPPORTED_PROVIDER"}

    try:
        r = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if r.status_code == 200:
            return {"valid": True}
        elif r.status_code == 401:
            return {"valid": False, "reason": "INVALID_KEY"}
        elif r.status_code == 403:
            return {"valid": False, "reason": "NO_PERMISSION"}
        elif r.status_code == 429:
            logger.info(f"{provider.capitalize()} Key Check Rate Limit: {r.status_code}")
            return {"valid": True, "reason": "RATE_LIMIT"} # Rate Limit이어도 키 자체는 유효함
        else:
            # Gemini의 경우 잘못된 키는 400 Bad Request를 반환할 수 있음
            if provider == "gemini" and r.status_code == 400:
                 return {"valid": False, "reason": "INVALID_KEY"}
                 
            logger.warning(f"Unknown status from {provider}: {r.status_code} - {r.text}")
            return {"valid": False, "reason": f"UNKNOWN_ERROR_{r.status_code}"}
            
    except Exception as e:
        logger.error(f"{provider.capitalize()} Key Check Error: {e}")
        return {"valid": False, "reason": "CONNECTION_ERROR"}