# DB Backup : 데이터 전용 SQL 백업 후 재적재(간단)

## 1. 데이터 전용 백업 파일 만들기
```py
# 백업 디렉터리 생성성
mkdir -p /home/work/CoreIQ/backend/storage/backup

# 전체 INSERT만 추출해 데이터 전용 SQL 생성
sqlite3 /home/work/CoreIQ/backend/storage/pps_rag.db ".dump" | grep -E '^INSERT INTO' \
  > /home/work/CoreIQ/backend/storage/backup/pps_rag_data_$(date +%F_%H%M%S).sql

# 파일 스냅샷(.backup)도 함께 보관(혹시 몰라서)
sqlite3 /home/work/CoreIQ/backend/storage/pps_rag.db ".backup '/home/work/CoreIQ/backend/storage/backup/pps_rag_$(date +%F_%H%M%S).db'"
```

## 2. 스키마 수정 후 새 DB 생성
```py
rm -f /home/work/CoreIQ/backend/storage/pps_rag.db
cd /home/work/CoreIQ/backend && python -c "import storage.database"
```

## 3. 데이터 재 업로드

```py
# 외래키 검사를 잠시 끄고 적재(테이블 순서 이슈 회피)
sqlite3 /home/work/CoreIQ/backend/storage/pps_rag.db "PRAGMA foreign_keys=OFF;"
sqlite3 /home/work/CoreIQ/backend/storage/pps_rag.db < /home/work/CoreIQ/backend/storage/backup/pps_rag_data_{YYYY-MM-DD_HHMMSS}.sql
sqlite3 /home/work/CoreIQ/backend/storage/pps_rag.db "PRAGMA foreign_keys=ON; PRAGMA foreign_key_check;"
```