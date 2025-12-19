from typing import Optional, Dict, Any
from sqlalchemy import select, update, func, text
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone, timedelta
import json
from utils import logger

from utils.database import get_session
from storage.db_models import FineTuneJob, FineTunedModel, LlmModel

log = logger(__name__)


def _now_kst() -> datetime:
    """KST(UTC+9) 기준 현재 시간 반환."""
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst)

def update_job_status(
    job_id: str, 
    status: str, 
    progress: Optional[int] = None, 
    extras: Optional[Dict[str, Any]] = None,
    rough: Optional[int] = None
) -> None:
    """파인튜닝 작업 상태 및 진행률 업데이트"""
    with get_session() as session:
        stmt = select(FineTuneJob).where(FineTuneJob.provider_job_id == job_id)
        job = session.execute(stmt).scalars().first()
        
        if not job:
            return

        job.status = status
        
        # metrics JSON 업데이트
        current_metrics = {}
        if job.metrics:
            try:
                current_metrics = json.loads(job.metrics)
            except:
                pass

        if progress is not None:
            # 내부 저장용 키 + API에서 사용하는 키를 모두 채워준다.
            # - progress            : 내부에서 과거에 쓰던 키
            # - learningProgress    : API 응답 스키마에서 사용하는 키
            current_metrics["progress"] = progress
            current_metrics["learningProgress"] = progress

        if rough is not None:
            # rough_score(과거) / roughScore(API 응답 키) 모두 기록
            current_metrics["rough_score"] = rough
            current_metrics["roughScore"] = rough

        if extras:
            current_metrics.update(extras)
            
        job.metrics = json.dumps(current_metrics, ensure_ascii=False)
        
        if status in ("succeeded", "failed"):
            # 작업 완료 시간도 KST 기준으로 기록
            job.finished_at = _now_kst()
            
        session.commit()

def finish_job_success(
    job_id: str,
    save_name: str,
    category: str,
    tuning_type: str,
    rouge_score: Optional[float],
    subcategory: Optional[str] = None
) -> None:
    """작업 성공 처리 및 FineTunedModel, LlmModel 등록"""
    with get_session() as session:
        # 1. Job 조회
        stmt = select(FineTuneJob).where(FineTuneJob.provider_job_id == job_id)
        job = session.execute(stmt).scalars().first()
        if not job:
            return

        # 2. 상태 완료 처리
        update_job_status(job_id, "succeeded", progress=100)

        # 3. LlmModel 등록/업데이트 (Fine-tuned 모델로) - Upsert 패턴
        # 기존 모델이 있으면 업데이트, 없으면 새로 생성
        existing_llm = session.execute(
            select(LlmModel).where(LlmModel.name == save_name)
        ).scalars().first()
        
        if existing_llm:
            # 기존 모델 업데이트
            log.info(f"Updating existing LlmModel: {save_name}")
            existing_llm.provider = "local"
            existing_llm.model_path = f"./storage/models/llm/{save_name}"
            existing_llm.category = category
            existing_llm.type = tuning_type.lower()
            existing_llm.is_active = True
            existing_llm.trained_at = _now_kst()
            new_llm = existing_llm
        else:
            # 새 모델 생성
            log.info(f"Creating new LlmModel: {save_name}")
            try:
                new_llm = LlmModel(
                    provider="local",  # 로컬 학습 모델
                    name=save_name,
                    revision=0,
                    model_path=f"./storage/models/llm/{save_name}", # 규칙에 따른 경로
                    category=category,
                    type=tuning_type.lower(), # lora, qlora, full
                    is_active=True,
                    trained_at=_now_kst()
                )
                session.add(new_llm)
                session.flush() # ID 생성/확인
            except IntegrityError as e:
                # PostgreSQL 시퀀스 동기화 문제 해결을 위한 예외 처리
                session.rollback()
                log.warning(f"LlmModel insert failed (likely sequence sync issue): {e}")
                
                # 다시 한번 기존 레코드 찾기
                existing_llm = session.execute(
                    select(LlmModel).where(LlmModel.name == save_name)
                ).scalars().first()
                if existing_llm:
                    log.info(f"Found existing LlmModel with id={existing_llm.id} for name={save_name}")
                    existing_llm.provider = "local"
                    existing_llm.model_path = f"./storage/models/llm/{save_name}"
                    existing_llm.category = category
                    existing_llm.type = tuning_type.lower()
                    existing_llm.is_active = True
                    existing_llm.trained_at = _now_kst()
                    new_llm = existing_llm
                else:
                    # 그래도 없으면 최대 ID를 찾아서 수동으로 생성
                    try:
                        # 현재 최대 ID 확인
                        max_id = session.execute(select(func.max(LlmModel.id))).scalar() or 0
                        next_id = max_id + 1
                        
                        # ID를 명시적으로 설정해서 재시도
                        new_llm = LlmModel(
                            id=next_id,
                            provider="local",
                            name=save_name,
                            revision=0,
                            model_path=f"./storage/models/llm/{save_name}",
                            category=category,
                            type=tuning_type.lower(),
                            is_active=True,
                            trained_at=_now_kst()
                        )
                        session.add(new_llm)
                        session.flush()
                        
                        # 시퀀스를 다음 ID로 안전하게 업데이트
                        session.execute(text(f"SELECT setval('llm_models_id_seq', {next_id}, true);"))
                        log.info(f"Updated llm_models_id_seq to next value: {next_id + 1}")
                        log.info(f"Created LlmModel with manually assigned id={next_id}")
                    except Exception as e2:
                        session.rollback()
                        log.exception(f"Failed to create LlmModel with manual ID: {e2}")
                        raise

        # 4. FineTunedModel 등록/업데이트 - Upsert 패턴
        # 같은 job_id로 이미 등록된 FineTunedModel이 있는지 확인
        existing_ft = session.execute(
            select(FineTunedModel).where(FineTunedModel.job_id == job.id)
        ).scalars().first()
        
        if existing_ft:
            # 기존 FineTunedModel 업데이트
            log.info(f"Updating existing FineTunedModel for job: {job_id}")
            existing_ft.model_id = new_llm.id
            existing_ft.provider_model_id = save_name
            existing_ft.lora_weights_path = f"./storage/models/llm/{save_name}" if tuning_type in ("LORA", "QLORA") else None
            existing_ft.type = tuning_type
            existing_ft.rouge1_f1 = rouge_score
            existing_ft.is_active = True
        else:
            # 새 FineTunedModel 생성
            log.info(f"Creating new FineTunedModel for job: {job_id}")
            ft_model = FineTunedModel(
                model_id=new_llm.id,
                job_id=job.id,
                provider_model_id=save_name,
                lora_weights_path=f"./storage/models/llm/{save_name}" if tuning_type in ("LORA", "QLORA") else None,
                type=tuning_type,
                rouge1_f1=rouge_score,
                is_active=True
            )
            session.add(ft_model)
        
        try:
            session.commit()
            log.info(f"Successfully registered/updated model: {save_name} for job: {job_id}")
        except IntegrityError as e:
            session.rollback()
            log.exception(f"Failed to register model {save_name}: {e}")
            # 재시도: 기존 모델 강제 업데이트
            try:
                existing_llm = session.execute(
                    select(LlmModel).where(LlmModel.name == save_name)
                ).scalars().first()
                if existing_llm:
                    existing_llm.model_path = f"./storage/models/llm/{save_name}"
                    existing_llm.category = category
                    existing_llm.type = tuning_type.lower()
                    existing_llm.is_active = True
                    existing_llm.trained_at = _now_kst()
                    session.commit()
                    log.warning(f"Force-updated existing model: {save_name}")
            except Exception as e2:
                log.exception(f"Force update also failed: {e2}")
                raise

def fail_job(job_id: str, error_msg: str) -> None:
    """작업 실패 처리"""
    update_job_status(job_id, "failed", extras={"error": error_msg})
