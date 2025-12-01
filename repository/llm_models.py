from typing import Any, Optional, Dict, List
from utils import logger
from utils.database import get_session
from sqlalchemy import select, or_, desc, func, update
from storage.db_models import (
    LlmModel, LlmPromptMapping, CacheData, SystemPromptTemplate,
    FineTunedModel, SystemPromptVariable, PromptMapping
)

logger = logger(__name__)


def get_llm_model_by_provider_and_name(provider: str, name: str) -> Optional[Dict[str, Any]]:
    """활성화된 LLM 모델을 provider와 name으로 조회한다."""
    logger.info(f"provider : {provider}\nmodelName : {name}")
    with get_session() as session:
        stmt = (
            select(LlmModel.provider, LlmModel.name, LlmModel.model_path)
            .where(
                LlmModel.name == name,
                LlmModel.provider == provider,
                LlmModel.is_active == True,
            )
            .limit(1)
        )
        row = session.execute(stmt).first()
        if row is None:
            logger.info("DB에서 가져온 모델 정보: None")
            return None
        try:
            data = dict(row._mapping)
        except Exception:
            logger.exception("row dict 변환 실패")
            return None
        logger.info(f"DB에서 가져온 모델 정보: {data}")
        return data


def repo_find_fallback_model(category: str) -> Optional[str]:
    """
    해당 카테고리(또는 all)의 활성화된 최신 모델 이름을 조회합니다.
    """
    with get_session() as session:
        stmt = select(LlmModel.name).where(
            LlmModel.is_active == True,
            or_(LlmModel.category == category, LlmModel.category == 'all')
        ).order_by(
            desc(LlmModel.trained_at), desc(LlmModel.id)
        ).limit(1)
        
        return session.execute(stmt).scalar()


def repo_find_best_mapped_model(prompt_id: int) -> Optional[str]:
    """
    템플릿(prompt_id)에 매핑된 모델 중 Rouge Score가 가장 높은(또는 최신) 모델 이름을 조회합니다.
    """
    with get_session() as session:
        stmt = select(LlmModel.name).join(
            LlmPromptMapping, LlmPromptMapping.llm_id == LlmModel.id
        ).where(
            LlmPromptMapping.prompt_id == prompt_id
        ).order_by(
            desc(func.coalesce(LlmPromptMapping.rouge_score, -1)),
            desc(LlmPromptMapping.llm_id)
        ).limit(1)
        
        return session.execute(stmt).scalar()


def repo_set_cache(name: str, data: str, belongs_to: str = "global", by_id: Optional[int] = None) -> None:
    """
    cache_data 테이블에 캐시 데이터를 저장합니다.
    """
    with get_session() as session:
        try:
            cache_data = CacheData(
                name=name,
                data=data,
                belongs_to=belongs_to,
                by_id=by_id
            )
            session.add(cache_data)
            session.commit()
            logger.debug(f"Cache data saved: name={name}, belongs_to={belongs_to}")
        except Exception:
            session.rollback()
            logger.exception(f"Failed to save cache data: name={name}")
            raise


def repo_get_cache(name: str) -> Optional[str]:
    """
    cache_data 테이블에서 name으로 최신 캐시 데이터를 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(CacheData.data)
                .where(CacheData.name == name)
                .order_by(desc(CacheData.id))
                .limit(1)
            )
            result = session.execute(stmt).scalar()
            return result
        except Exception:
            logger.exception(f"Failed to get cache data: name={name}")
            return None


def repo_get_llm_model_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """
    llm_models 테이블에서 name으로 모델을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = select(LlmModel).where(LlmModel.name == model_name).limit(1)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                return None
            return {
                "id": model.id,
                "provider": model.provider,
                "name": model.name,
                "revision": model.revision,
                "model_path": model.model_path,
                "category": model.category,
                "mather_path": model.mather_path,
                "type": model.type,
                "is_default": model.is_default,
                "is_active": model.is_active,
                "trained_at": model.trained_at,
                "created_at": model.created_at,
            }
        except Exception:
            logger.exception(f"Failed to get llm model by name: {model_name}")
            return None


def repo_get_active_llm_models() -> List[Dict[str, Any]]:
    """
    활성화된(is_active=1) LLM 모델 목록을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    LlmModel.id,
                    LlmModel.provider,
                    LlmModel.name,
                    LlmModel.category,
                    LlmModel.model_path,
                    LlmModel.type,
                    LlmModel.is_default,
                    LlmModel.is_active,
                    LlmModel.trained_at,
                    LlmModel.created_at,
                )
                .where(LlmModel.is_active == True)
                .order_by(desc(LlmModel.id))
            )
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception("Failed to get active llm models")
            return []


def repo_update_llm_model_active_by_path(model_path: str, is_active: bool) -> None:
    """
    model_path로 llm_models의 is_active를 업데이트합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                update(LlmModel)
                .where(LlmModel.model_path == model_path)
                .values(is_active=is_active)
            )
            session.execute(stmt)
            session.commit()
            logger.debug(f"Updated llm model is_active: path={model_path}, is_active={is_active}")
        except Exception:
            session.rollback()
            logger.exception(f"Failed to update llm model is_active: path={model_path}")
            raise


def repo_get_llm_model_path_by_name(model_name: str) -> Optional[str]:
    """
    llm_models 테이블에서 name으로 model_path를 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(LlmModel.model_path)
                .where(LlmModel.name == model_name)
                .limit(1)
            )
            return session.execute(stmt).scalar()
        except Exception:
            logger.exception(f"Failed to get llm model path by name: {model_name}")
            return None


def repo_get_llm_model_id_and_path_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """
    llm_models 테이블에서 name으로 id와 model_path를 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(LlmModel.id, LlmModel.model_path)
                .where(LlmModel.name == model_name)
                .limit(1)
            )
            row = session.execute(stmt).first()
            if row is None:
                return None
            return {"id": row.id, "model_path": row.model_path}
        except Exception:
            logger.exception(f"Failed to get llm model id and path by name: {model_name}")
            return None


def repo_get_llm_models_by_category_all() -> List[Dict[str, Any]]:
    """
    category=all인 경우 전체 모델 목록을 조회합니다 (활성/비활성 포함).
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    LlmModel.id,
                    LlmModel.name,
                    LlmModel.provider,
                    LlmModel.category,
                    LlmModel.is_active.label("isActive"),
                    LlmModel.trained_at,
                    LlmModel.created_at,
                )
                .order_by(
                    desc(LlmModel.is_active),
                    desc(LlmModel.trained_at),
                    desc(LlmModel.id)
                )
            )
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception("Failed to get llm models by category all")
            return []


def repo_get_llm_models_by_category_and_subcategory(
    category: str, subcategory: str
) -> List[Dict[str, Any]]:
    """
    doc_gen + subcategory인 경우 매핑된 모델을 rouge 점수순으로 조회합니다.
    """
    with get_session() as session:
        try:
            # 먼저 prompt template id를 찾습니다
            prompt_stmt = (
                select(SystemPromptTemplate.id)
                .where(
                    SystemPromptTemplate.category == category,
                    SystemPromptTemplate.name == subcategory,
                    or_(
                        SystemPromptTemplate.is_active == True,
                        SystemPromptTemplate.is_active.is_(None)
                    )
                )
                .order_by(
                    desc(func.coalesce(SystemPromptTemplate.is_default, False)),
                    desc(SystemPromptTemplate.id)
                )
                .limit(1)
            )
            prompt_id = session.execute(prompt_stmt).scalar()
            
            if prompt_id is None:
                return []
            
            # prompt_id로 매핑된 모델들을 조회
            stmt = (
                select(
                    LlmModel.id,
                    LlmModel.name,
                    LlmModel.provider,
                    LlmModel.category,
                    LlmModel.is_active.label("isActive"),
                    LlmModel.trained_at,
                    LlmModel.created_at,
                    func.coalesce(LlmPromptMapping.rouge_score, -1).label("rougeScore"),
                )
                .join(LlmPromptMapping, LlmPromptMapping.llm_id == LlmModel.id)
                .where(
                    LlmPromptMapping.prompt_id == prompt_id,
                    or_(
                        LlmModel.category == category,
                        LlmModel.category == 'all'
                    )
                )
                .order_by(
                    desc(func.coalesce(LlmPromptMapping.rouge_score, -1)),
                    desc(LlmModel.trained_at),
                    desc(LlmModel.id)
                )
            )
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception(f"Failed to get llm models by category and subcategory: {category}, {subcategory}")
            return []


def repo_get_llm_models_by_category(category: str) -> List[Dict[str, Any]]:
    """
    특정 category의 활성화된 모델 목록을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    LlmModel.id,
                    LlmModel.name,
                    LlmModel.provider,
                    LlmModel.category,
                    LlmModel.is_active.label("isActive"),
                    LlmModel.trained_at,
                    LlmModel.created_at,
                )
                .where(
                    LlmModel.is_active == True,
                    or_(
                        LlmModel.category == category,
                        LlmModel.category == 'all'
                    )
                )
                .order_by(desc(LlmModel.trained_at), desc(LlmModel.id))
            )
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception(f"Failed to get llm models by category: {category}")
            return []


def repo_delete_llm_model(model_id: int) -> int:
    """
    llm_models 테이블에서 id로 모델을 삭제합니다.
    삭제된 행 수를 반환합니다.
    """
    with get_session() as session:
        try:
            stmt = select(LlmModel).where(LlmModel.id == model_id)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                return 0
            session.delete(model)
            session.commit()
            logger.info(f"Deleted llm model: id={model_id}")
            return 1
        except Exception:
            session.rollback()
            logger.exception(f"Failed to delete llm model: id={model_id}")
            raise


def repo_get_llm_model_by_id(model_id: int) -> Optional[Dict[str, Any]]:
    """
    llm_models 테이블에서 id로 모델을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = select(LlmModel).where(LlmModel.id == model_id).limit(1)
            model = session.execute(stmt).scalar_one_or_none()
            if model is None:
                return None
            return {
                "id": model.id,
                "name": model.name,
                "provider": model.provider,
                "type": model.type,
                "model_path": model.model_path,
                "mather_path": model.mather_path,
                "category": model.category,
                "is_active": model.is_active,
                "trained_at": model.trained_at,
                "created_at": model.created_at,
            }
        except Exception:
            logger.exception(f"Failed to get llm model by id: {model_id}")
            return None


def repo_get_fine_tuned_model_by_model_id(model_id: int, active_first: bool = True) -> Optional[Dict[str, Any]]:
    """
    fine_tuned_models 테이블에서 model_id로 파인튜닝 모델을 조회합니다.
    active_first=True이면 활성 모델 우선, 없으면 최신 모델을 반환합니다.
    """
    with get_session() as session:
        try:
            if active_first:
                # 활성 모델 우선 조회
                stmt = (
                    select(FineTunedModel)
                    .where(
                        FineTunedModel.model_id == model_id,
                        or_(
                            FineTunedModel.is_active == True,
                            FineTunedModel.is_active.is_(None)
                        )
                    )
                    .order_by(desc(FineTunedModel.id))
                    .limit(1)
                )
                ft = session.execute(stmt).scalar_one_or_none()
                if ft:
                    return {
                        "id": ft.id,
                        "model_id": ft.model_id,
                        "job_id": ft.job_id,
                        "provider_model_id": ft.provider_model_id,
                        "lora_weights_path": ft.lora_weights_path,
                        "base_model_id": ft.base_model_id,
                        "base_model_path": ft.base_model_path,
                        "rouge1_f1": ft.rouge1_f1,
                        "type": ft.type,
                        "is_active": ft.is_active,
                        "created_at": ft.created_at,
                    }
            
            # 활성 모델이 없거나 active_first=False인 경우 최신 모델 조회
            stmt = (
                select(FineTunedModel)
                .where(FineTunedModel.model_id == model_id)
                .order_by(desc(FineTunedModel.id))
                .limit(1)
            )
            ft = session.execute(stmt).scalar_one_or_none()
            if ft:
                return {
                    "id": ft.id,
                    "model_id": ft.model_id,
                    "job_id": ft.job_id,
                    "provider_model_id": ft.provider_model_id,
                    "lora_weights_path": ft.lora_weights_path,
                    "base_model_id": ft.base_model_id,
                    "base_model_path": ft.base_model_path,
                    "rouge1_f1": ft.rouge1_f1,
                    "type": ft.type,
                    "is_active": ft.is_active,
                    "created_at": ft.created_at,
                }
            return None
        except Exception:
            logger.exception(f"Failed to get fine tuned model by model_id: {model_id}")
            return None


def repo_get_distinct_model_ids_from_fine_tuned_models() -> List[int]:
    """
    fine_tuned_models 테이블에서 고유한 model_id 목록을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = select(func.distinct(FineTunedModel.model_id)).where(
                FineTunedModel.model_id.isnot(None)
            )
            rows = session.execute(stmt).all()
            return [int(row[0]) for row in rows if row[0] is not None]
        except Exception:
            logger.exception("Failed to get distinct model_ids from fine_tuned_models")
            return []


def repo_get_prompt_template_by_id(prompt_id: int) -> Optional[Dict[str, Any]]:
    """
    system_prompt_template 테이블에서 id로 활성화된 템플릿을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(SystemPromptTemplate)
                .where(
                    SystemPromptTemplate.id == prompt_id,
                    or_(
                        SystemPromptTemplate.is_active == True,
                        SystemPromptTemplate.is_active.is_(None)
                    )
                )
                .limit(1)
            )
            template = session.execute(stmt).scalar_one_or_none()
            if template is None:
                return None
            return {
                "id": template.id,
                "name": template.name,
                "category": template.category,
                "system_prompt": template.system_prompt,
                "user_prompt": template.user_prompt,
                "is_default": template.is_default,
                "is_active": template.is_active,
            }
        except Exception:
            logger.exception(f"Failed to get prompt template by id: {prompt_id}")
            return None


def repo_get_prompt_variables_by_template_id(template_id: int) -> List[Dict[str, Any]]:
    """
    system_prompt_variables 테이블에서 template_id로 연결된 변수들을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    SystemPromptVariable.id,
                    SystemPromptVariable.type,
                    SystemPromptVariable.key,
                    SystemPromptVariable.value,
                    SystemPromptVariable.description,
                )
                .join(PromptMapping, PromptMapping.variable_id == SystemPromptVariable.id)
                .where(PromptMapping.template_id == template_id)
            )
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception(f"Failed to get prompt variables by template_id: {template_id}")
            return []


def repo_get_prompt_templates_by_category_and_name(
    category: str, name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    system_prompt_template 테이블에서 category와 name(선택)으로 활성화된 템플릿들을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    SystemPromptTemplate.id,
                    SystemPromptTemplate.name,
                    SystemPromptTemplate.system_prompt,
                    SystemPromptTemplate.user_prompt,
                )
                .where(
                    SystemPromptTemplate.category == category,
                    or_(
                        SystemPromptTemplate.is_active == True,
                        SystemPromptTemplate.is_active.is_(None)
                    )
                )
            )
            if name:
                stmt = stmt.where(func.lower(SystemPromptTemplate.name) == name.lower())
            stmt = stmt.order_by(desc(SystemPromptTemplate.id))
            rows = session.execute(stmt).all()
            return [dict(row._mapping) for row in rows]
        except Exception:
            logger.exception(f"Failed to get prompt templates by category and name: {category}, {name}")
            return []


def repo_get_default_prompt_template_by_category(
    category: str, subcategory: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    system_prompt_template 테이블에서 category와 subcategory(선택)로 기본 템플릿을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    SystemPromptTemplate.id,
                    SystemPromptTemplate.name,
                )
                .where(
                    SystemPromptTemplate.category == category,
                    func.coalesce(SystemPromptTemplate.is_default, False) == True,
                    or_(
                        SystemPromptTemplate.is_active == True,
                        SystemPromptTemplate.is_active.is_(None)
                    )
                )
            )
            if subcategory:
                stmt = stmt.where(func.lower(SystemPromptTemplate.name) == subcategory.lower())
            stmt = stmt.order_by(desc(SystemPromptTemplate.id)).limit(1)
            row = session.execute(stmt).first()
            if row:
                return {"id": row.id, "name": row.name}
            return None
        except Exception:
            logger.exception(f"Failed to get default prompt template: {category}, {subcategory}")
            return None


def repo_get_best_llm_prompt_mapping_by_prompt_id(prompt_id: int) -> Optional[Dict[str, Any]]:
    """
    llm_prompt_mapping 테이블에서 prompt_id로 가장 높은 rouge_score를 가진 매핑을 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(
                    LlmPromptMapping.llm_id,
                    LlmPromptMapping.prompt_id,
                    LlmPromptMapping.rouge_score,
                )
                .where(LlmPromptMapping.prompt_id == prompt_id)
                .order_by(
                    desc(func.coalesce(LlmPromptMapping.rouge_score, -1)),
                    desc(LlmPromptMapping.llm_id)
                )
                .limit(1)
            )
            row = session.execute(stmt).first()
            if row:
                return {
                    "llm_id": row.llm_id,
                    "prompt_id": row.prompt_id,
                    "rouge_score": row.rouge_score,
                }
            return None
        except Exception:
            logger.exception(f"Failed to get best llm prompt mapping by prompt_id: {prompt_id}")
            return None


def repo_count_default_prompt_templates(
    category: str, subcategory: Optional[str] = None
) -> int:
    """
    system_prompt_template 테이블에서 category와 subcategory(선택)로 기본 템플릿 개수를 조회합니다.
    """
    with get_session() as session:
        try:
            stmt = (
                select(func.count(SystemPromptTemplate.id))
                .where(
                    SystemPromptTemplate.category == category,
                    func.coalesce(SystemPromptTemplate.is_default, False) == True,
                    or_(
                        SystemPromptTemplate.is_active == True,
                        SystemPromptTemplate.is_active.is_(None)
                    )
                )
            )
            if subcategory:
                stmt = stmt.where(func.lower(SystemPromptTemplate.name) == subcategory.lower())
            result = session.execute(stmt).scalar()
            return result or 0
        except Exception:
            logger.exception(f"Failed to count default prompt templates: {category}, {subcategory}")
            return 0