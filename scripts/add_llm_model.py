#python /home/work/CoreIQ/backend/scripts/add_llm_model.py
from utils import get_db, logger
log = logger(__name__)

def add_llm_model():
    row = {
        "provider":  "huggingface",
        "name":      "local_gpt_oss_20b",
        "revision":  0,
        "model_path":"./service/storage/model/local_gpt_oss_20b",
        "category":  "qa",
        "type":      "base",
        "is_default":0,
        "is_active": 1,
    }
    sql = """
        INSERT INTO llm_models
          (provider,name,revision,model_path,category,type,is_default,is_active,trained_at)
        VALUES
          (:provider,:name,:revision,:model_path,:category,:type,:is_default,:is_active,CURRENT_TIMESTAMP)
    """
    conn = get_db()
    try:
        conn.execute(sql, row)
        conn.commit()
        log.info("llm_model inserted!")
    except Exception as e:
        log.error(f"failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_llm_model()