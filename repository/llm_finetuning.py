from typing import Optional, Dict, Any
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from datetime import datetime
import json

from utils.database import get_session
from storage.db_models import FineTuneJob, FineTunedModel, LlmModel

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
            current_metrics["progress"] = progress
        
        if rough is not None:
             current_metrics["rough_score"] = rough

        if extras:
            current_metrics.update(extras)
            
        job.metrics = json.dumps(current_metrics, ensure_ascii=False)
        
        if status in ("succeeded", "failed"):
            job.finished_at = datetime.utcnow()
            
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

        # 3. LlmModel 등록 (Fine-tuned 모델로)
        # category는 job 요청정보나 파라미터에서 옴
        new_llm = LlmModel(
            provider="local",  # 로컬 학습 모델
            name=save_name,
            revision=0,
            model_path=f"./storage/models/{save_name}", # 규칙에 따른 경로
            category=category,
            type=tuning_type.lower(), # lora, qlora, full
            is_active=True,
            trained_at=datetime.utcnow()
        )
        session.add(new_llm)
        session.flush() # ID 생성

        # 4. FineTunedModel 등록
        # base_model_id 등을 job 정보에서 가져와야 하지만, 
        # 여기서는 job.dataset_id 등을 통해 역추적하거나, 인자로 받아야 함.
        # 기존 로직 상 baseModelName을 request에서 썼음.
        
        # request 정보 파싱
        req = {}
        # Job 테이블에 request 컬럼이 없어서(DB모델상), 
        # 기존 코드는 메모리의 job 객체(Pydantic)를 썼음.
        # 여기서는 DB Job 레코드만으로는 부족할 수 있으나, 
        # FineTunedModel 연결을 위해 최소한의 정보를 저장
        
        ft_model = FineTunedModel(
            model_id=new_llm.id,
            job_id=job.id,
            provider_model_id=save_name,
            lora_weights_path=f"./storage/models/{save_name}" if tuning_type in ("LORA", "QLORA") else None,
            type=tuning_type,
            rouge1_f1=rouge_score,
            is_active=True
        )
        session.add(ft_model)
        session.commit()

def fail_job(job_id: str, error_msg: str) -> None:
    """작업 실패 처리"""
    update_job_status(job_id, "failed", extras={"error": error_msg})
