BEGIN TRANSACTION;

-- 1) config 테이블 재정의(서비스별 복수행 허용: 복합 PK)
ALTER TABLE security_level_config_task RENAME TO _slc_old;

CREATE TABLE security_level_config_task (
  task_type TEXT NOT NULL,                          -- 'doc_gen' | 'summary' | 'qna'
  service   TEXT NOT NULL DEFAULT 'global',         -- 서비스 이름(드롭다운 값)
  max_level INTEGER NOT NULL,
  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (task_type, service)                  -- ★ 복합 PK로 변경
);

-- 기존 데이터는 'global' 서비스로 이관
INSERT INTO security_level_config_task(task_type, service, max_level, updated_at)
SELECT task_type, 'global', max_level, updated_at
FROM _slc_old;

DROP TABLE _slc_old;

-- 2) keywords 테이블에 service 컬럼 추가
ALTER TABLE security_level_keywords_task
  ADD COLUMN service TEXT NOT NULL DEFAULT 'global';

-- 3) 조회/중복 방지 인덱스
CREATE INDEX IF NOT EXISTS ix_slk_task_level_service
  ON security_level_keywords_task(task_type, level, service, keyword);

-- (선택) 동일 서비스·레벨 내 중복 키워드 방지
CREATE UNIQUE INDEX IF NOT EXISTS ux_slk_unique
  ON security_level_keywords_task(task_type, level, keyword, service);

COMMIT;

