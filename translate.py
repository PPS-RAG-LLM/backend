import sqlite3, pathlib

def get_db():
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def set_default_llm_model(name: str, category: str ) -> bool:
    """
    - name으로 대상 모델 조회
    - (옵션) category가 주어지면 대상 모델의 category와 일치 검증
    - 같은 category에서 대상만 is_default=1, 나머지는 0으로 원샷 토글
    - 비활성(is_active=0) 모델은 기본값 불가
    """
    con = get_db()
    try:
        cur = con.cursor()
        cur.execute("SELECT id, category, is_active FROM llm_models WHERE name=?", (name,))
        row = cur.fetchone()
        print(row)
        if not row:
            print(f"[WARN] LLM model not found: name={name}")
            return False

        model_id = row["id"]
        model_category = row["category"]
        is_active = bool(row["is_active"])

        if category and category != model_category:
            raise ValueError(f"category mismatch: expected '{category}', got '{model_category}' for model '{name}'")
        if not is_active:
            raise ValueError(f"cannot set inactive model as default: name={name}")

        # 같은 카테고리에서 대상만 1, 나머지는 0
        cur.execute(
            """
            UPDATE llm_models
            SET is_default = CASE WHEN id = ? THEN 1 ELSE 0 END
            WHERE category = ?
            """,
            (model_id, model_category),
        )
        con.commit()
        print(f"[OK] Default LLM set: category={model_category}, name={name}")
        return cur.rowcount > 0
    except sqlite3.IntegrityError as e:
        con.rollback()
        raise

if __name__ == "__main__":
    path = "/home/work/CoreIQ/Ruah/backend/storage/pps_rag.db"
    schema_path = "/home/work/CoreIQ/Ruah/backend/storage/schema.sql"

    sql = pathlib.Path(schema_path).read_text(encoding="utf-8")
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON;")
    con.executescript(sql)
    con.commit()
    con.close()

    set_default_llm_model("gpt_oss_20b", "qa")