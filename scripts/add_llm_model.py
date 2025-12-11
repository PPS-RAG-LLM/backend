#python scripts/add_llm_model.py
from utils.database import get_session
from utils import logger
from storage.db_models import LlmModel
from datetime import datetime

log = logger(__name__)

def add_llm_model():
    # ORM 객체 생성
    new_model = LlmModel(
        provider="huggingface",
        name="local_gpt_oss_20b",
        revision=0,
        model_path="./storage/models/llm/local_gpt_oss_20b",
        category="qna",
        type="base",
        is_active=True,  # Postgres boolean 호환 (True/False 사용)
        trained_at=datetime.utcnow()
    )

    session = get_session()
    try:
        session.add(new_model)
        session.commit()
        log.info(f"llm_model '{new_model.name}' inserted!")
    except Exception as e:
        session.rollback()
        log.error(f"failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    add_llm_model()
