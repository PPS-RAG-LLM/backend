"""
RAG 전처리 라우터 모듈
확장자별로 적절한 전처리 함수를 호출하는 라우팅 기능 제공
"""
from __future__ import annotations

import logging
import shutil
from collections import defaultdict

from pathlib import Path
from typing import Any, List, Optional

from tqdm import tqdm  # type: ignore

from repository.documents import bulk_upsert_document_metadata, delete_documents_not_in_doc_ids
from utils.documents import generate_doc_id
from service.retrieval.common import determine_level_for_task, parse_doc_version

from service.preprocessing.extension.csv_preprocessing import extract_csv
from service.preprocessing.extension.docx_preprocessing import extract_docx
from service.preprocessing.extension.excel_preprocessing import extract_excel
from service.preprocessing.extension.hwp_preprocessing import extract_hwp
from service.preprocessing.extension.pdf_preprocessing import extract_pdf_with_tables
from service.preprocessing.extension.ppt_preprocessing import extract_ppt
from service.preprocessing.extension.pptx_preprocessing import extract_pptx
from service.preprocessing.extension.txt_preprocessing import extract_plain_text
from utils.time import now_kst_string

logger = logging.getLogger(__name__)


def ext(p: Path) -> str:
    """파일 확장자 반환 (소문자)"""
    return p.suffix.lower()


def _clean_text(s: str | None) -> str:
    """텍스트 정규화 (공통 유틸리티에서 import)"""
    from service.preprocessing.extension.utils import clean_text as clean_text_util

    return clean_text_util(s)


def extract_any(path: Path) -> tuple[str, list[dict]]:
    """통합 문서 추출 라우터"""
    file_ext = ext(path)
    if file_ext == ".pdf":
        text, tables, _, _ = extract_pdf_with_tables(path)
        return text, tables
    if file_ext in {".txt", ".text", ".md"}:
        return extract_plain_text(path)
    if file_ext == ".docx":
        return extract_docx(path)
    if file_ext == ".pptx":
        return extract_pptx(path)
    if file_ext == ".csv":
        return extract_csv(path)
    if file_ext in {".xlsx", ".xls"}:
        return extract_excel(path)
    if ext == ".ppt":
        return extract_ppt(path)
    # DOC 파일은 DOCX로 변환 후 처리 (docx_preprocessing에서 자동 변환)
    if ext == ".doc":
        return extract_docx(path)
    # HWP 파일은 현재 지원하지 않음 (Windows 서버 필요)
    if ext == ".hwp":
        logger.warning(f"[Extract] HWP 파일은 현재 지원하지 않습니다: {path.name}")
        return "", []
    # 모르는 확장자는 텍스트로 시도
    return extract_plain_text(path)


async def extract_documents(target_rel_paths: Optional[List[str]] = None):
    """
    문서 전처리 메인 함수
    모든 확장자의 파일을 extract_any를 사용하여 처리합니다.
    
    Returns:
        dict: 전처리 결과
    """
    from service.admin.manage_vator_DB import (
        ADMIN_DOC_TYPE,
        RAW_DATA_DIR,
        TASK_TYPES,
        SUPPORTED_EXTS,
        get_security_level_rules_all,
        register_admin_document,
    )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 규칙 로드
    all_rules = get_security_level_rules_all()  # {task: {"maxLevel":N, "levels":{...}}}

    def _extract_base_and_date(p: Path):
        name = p.stem
        parts = name.split("_")
        date_num = 0
        if len(parts) >= 2:
            cand = parts[-1]
            if cand.isdigit() and len(cand) in (4, 6, 8):
                try:
                    date_num = int(cand)
                except Exception:
                    date_num = 0
        mid_tokens = [t for t in parts[:-1] if t and not t.isdigit()]
        base = max(mid_tokens, key=len) if mid_tokens else parts[0]
        return base, date_num

    raw_files = [p for p in RAW_DATA_DIR.rglob("*") if p.is_file() and ext(p) in SUPPORTED_EXTS]
    if target_rel_paths:
        wanted = set()
        for rel in target_rel_paths:
            candidate = (RAW_DATA_DIR / Path(rel)).resolve()
            if candidate.exists():
                wanted.add(candidate)
            else:
                logger.warning("[Extract] target file not found in RAW: %s", rel)
        raw_files = [p for p in raw_files if p.resolve() in wanted]
        if not raw_files:
            return {
                "message": "대상 RAW 파일을 찾지 못했습니다.",
                "document_count": 0,
                "file_count": 0,
                "processed_doc_ids": [],
            }

    grouped: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    for p in raw_files:
        base, date_num = _extract_base_and_date(p)
        grouped[base].append((p, date_num))

    kept, removed = [], []
    for base, lst in grouped.items():
        lst_sorted = sorted(
            lst,
            key=lambda it: (it[1], it[0].stat().st_mtime, len(it[0].name))  # date_num → mtime → 이름 길이
        )
        keep_path = lst_sorted[-1][0]
        kept.append(keep_path)
        for old_path, _ in lst_sorted[:-1]:
            try:
                old_path.unlink(missing_ok=True)
                removed.append(str(old_path.relative_to(RAW_DATA_DIR)))
            except Exception:
                logger.exception("Failed to remove duplicate: %s", old_path)

    if not kept:
        return {
            "message": "처리할 문서가 없습니다.",
            "document_count": 0,
            "deduplicated": {"removedCount": len(removed), "removed": removed},
            "processed_doc_ids": [],
        }

    processed_doc_ids: list[str] = []
    for src in tqdm(kept, desc="문서 전처리"):
        try:
            logger.info(f"[Extract] 파일 처리 시작: {src.name}")

            pages_text_dict: dict[int, str] = {}
            total_pages = 0

            if ext(src) == ".pdf":
                text, tables, pages_text_dict, total_pages = extract_pdf_with_tables(src)
            else:
                text, tables = extract_any(src)

            if tables:
                logger.info(f"[Extract] {src.name}: 표 {len(tables)}개 추출됨")
                for t_idx, t in enumerate(tables):
                    page = t.get("page", 0)
                    text_preview = (t.get("text", "")[:100] + "...") if t.get("text") else ""
                    logger.info(f"[Extract] {src.name} 표 {t_idx+1}: 페이지={page}, 텍스트 미리보기={text_preview}")
            else:
                logger.info(f"[Extract] {src.name}: 추출된 표 없음 (텍스트만 추출됨)")

            whole_for_level = text + "\n\n" + "\n\n".join(t.get("text","") for t in (tables or []))
            sec_map = {
                task: determine_level_for_task(
                    whole_for_level,
                    all_rules.get(task, {"maxLevel": 1, "levels": {}})
                )
                for task in TASK_TYPES
            }

            max_sec = max(sec_map.values()) if sec_map else 1
            sec_folder = f"securityLevel{int(max_sec)}"

            rel_from_raw = src.relative_to(RAW_DATA_DIR)
            rel_source_path = str(Path("row_data") / rel_from_raw.as_posix())  # RAW 기준 경로

            pages_tables: dict[int, list[dict]] = defaultdict(list)
            for t in (tables or []):
                page_num = t.get("page", 0)
                if page_num > 0:
                    pages_tables[page_num].append(t)

            metadata_records: list[dict[str, Any]] = []
            chunk_index = 0

            def _append_record(page: int, chunk_text: str, *, extra_payload: Optional[dict] = None):
                nonlocal chunk_index
                payload = {"source_file": src.name}
                if extra_payload:
                    payload.update(extra_payload)
                metadata_records.append(
                    {
                        "page": int(page),
                        "chunk_index": int(chunk_index),
                        "text": chunk_text,
                        "payload": payload,
                    }
                )
                chunk_index += 1

            if pages_text_dict:
                all_page_nums = sorted(set(pages_text_dict) | set(pages_tables))
                for page_num in all_page_nums:
                    page_text = pages_text_dict.get(page_num, "")
                    if page_text.strip():
                        _append_record(page_num, page_text)
                    for tbl in pages_tables.get(page_num, []):
                        table_text = tbl.get("text", "")
                        if table_text.strip():
                            _append_record(
                                page_num,
                                table_text,
                                extra_payload={"table": True, "table_bbox": tbl.get("bbox")}
                            )
            else:
                if text.strip():
                    _append_record(1, text)
                for tbl in tables or []:
                    table_text = tbl.get("text", "")
                    if table_text.strip():
                        _append_record(
                            int(tbl.get("page") or 0),
                            table_text,
                            extra_payload={"table": True, "table_bbox": tbl.get("bbox")}
                        )

            stem = rel_from_raw.stem
            _, version = parse_doc_version(stem)
            doc_id = generate_doc_id()

            preview = (_clean_text(text[:200].replace("\n"," ")) + "…") if text else ""
            extraction_info = {
                "original_file": src.name,
                "text_length": len(text),
                "table_count": len(tables or []),
                "extracted_at": now_kst_string(),
            }

            register_admin_document(
                doc_id=doc_id,
                filename=src.name,
                rel_text_path=rel_source_path,          # storage_path 대체로 RAW 경로 사용
                rel_source_path=rel_source_path,        # payload["saved_files"]["source"]도 동일
                sec_map=sec_map,
                version=int(version),
                preview=preview,
                tables=tables or [],
                total_pages=total_pages,
                pages=pages_text_dict if pages_text_dict else {},
                source_ext=ext(src),
                extraction_info=extraction_info,
            )
            if metadata_records:
                bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)

            processed_doc_ids.append(doc_id)
            logger.info(
                f"[Extract] {src.name}: 메타데이터 저장 완료 (텍스트={len(text)}자, 표={len(tables or [])}개)"
            )

        except Exception:
            logger.exception("Failed to process: %s", src)

    deleted_docs = 0
    if processed_doc_ids and not target_rel_paths:
        unique_ids = list(dict.fromkeys(processed_doc_ids))
        deleted_docs = delete_documents_not_in_doc_ids(ADMIN_DOC_TYPE, unique_ids)

    return {
        "message": "문서 추출 완료",
        "file_count": len(kept),
        "document_count": len(processed_doc_ids),
        "deleted_documents": int(deleted_docs),
        "deduplicated": {"removedCount": len(removed), "removed": removed},
        "processed_doc_ids": processed_doc_ids,
    }
