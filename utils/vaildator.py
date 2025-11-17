"""API 카테고리 요청 검증 유틸리티"""
from typing import Optional
from fastapi import Query
from errors import BadRequestError


def validate_category_subcategory(
    category: str = Query(..., pattern="^(qna|doc_gen|summary)$", description="카테고리"),
    subcategory: Optional[str] = Query(None, description="서브카테고리 (doc_gen: meeting, business_trip, report)")
) -> tuple[str, Optional[str]]:
    """
    category와 subcategory의 유효성을 검증합니다.
    
    Rules:
    - doc_gen: subcategory 필수 (meeting, business_trip, report)
    - qna, summary: subcategory 불가
    
    Returns:
        tuple[str, Optional[str]]: (category, subcategory)
    
    Raises:
        BadRequestError: 검증 실패 시
    """
    category = validate_category(category)
    
    if category == "doc_gen":
        if not subcategory:
            raise BadRequestError("문서 생성(doc_gen) 카테고리는 subcategory가 필수입니다.")
        
        allowed_subcategories = ["meeting", "business_trip", "report"]
        if subcategory not in allowed_subcategories:
            raise BadRequestError(
                f"doc_gen 카테고리의 subcategory는 {', '.join(allowed_subcategories)} 중 하나여야 합니다"
            )
    else:  # qna 또는 summary
        if subcategory:
            raise BadRequestError(f"{category} 카테고리는 subcategory를 사용할 수 없습니다")
        subcategory = None
    
    return category, subcategory

def validate_category(category: str = Query(..., pattern="^(qna|doc_gen|summary)$", description="카테고리")) -> str:
    """
    카테고리의 유효성을 검증합니다.
    """
    allowed_categories = ["qna", "doc_gen", "summary"]
    if category not in allowed_categories:
        raise BadRequestError(f"{category} 카테고리는 사용할 수 없습니다")
    return category