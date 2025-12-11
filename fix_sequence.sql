-- PostgreSQL 시퀀스 동기화 문제 해결
-- 이 스크립트는 fine_tune_datasets 테이블의 시퀀스를 올바르게 동기화합니다.

-- 1. 현재 테이블의 최대 ID 확인
SELECT MAX(id) FROM fine_tune_datasets;

-- 2. 시퀀스를 테이블의 최대 ID + 1로 설정
SELECT setval('fine_tune_datasets_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM fine_tune_datasets), false);

-- 3. 다른 테이블들도 동일하게 처리 (필요한 경우)
SELECT setval('fine_tune_jobs_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM fine_tune_jobs), false);
SELECT setval('fine_tuned_models_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM fine_tuned_models), false);
SELECT setval('llm_models_id_seq', (SELECT COALESCE(MAX(id), 0) + 1 FROM llm_models), false);

-- 4. 확인 (시퀀스 값 체크)
SELECT currval('fine_tune_datasets_id_seq');
