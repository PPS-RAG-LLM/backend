"""
PDF 전처리 모듈
PDF 파일을 텍스트와 표로 추출하는 기능 제공
"""
from __future__ import annotations
import shutil
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# now_kst_string은 utils.time에서 import
from utils.time import now_kst_string

# 텍스트 정리용 정규식
ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')
CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # \t,\n은 유지
MULTISPACE_LINE_END_RE = re.compile(r'[ \t]+\n')
NEWLINES_RE = re.compile(r'\n{3,}')


def ext(p: Path) -> str:
    """파일 확장자 반환 (소문자)"""
    return p.suffix.lower()


def _clean_text(s: str | None) -> str:
    """PDF 특수문자 제거 및 텍스트 정규화"""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # 흔한 공백/불릿/문장부호 정리
    s = s.replace("\xa0", " ").replace("\u2022", "•").replace("\u2212", "-")
    s = ZERO_WIDTH_RE.sub("", s)
    s = CONTROL_RE.sub("", s)
    s = MULTISPACE_LINE_END_RE.sub("\n", s)
    s = NEWLINES_RE.sub("\n\n", s)
    return s.strip()


def _df_to_markdown_repeat_header(df, max_rows=500) -> str:
    """
    각 데이터 행 앞에 헤더를 반복해서 출력하는 마크다운 테이블 생성.
    예)
      이름 | 나이 | 성
      기정 | 29  | 남자
      이름 | 나이 | 성
      수현 | 26  | 여자
    """
    import pandas as pd
    if isinstance(df, pd.DataFrame) and len(df) > max_rows:
        df = df.head(max_rows)

    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"

    lines = []
    for _, row in df.iterrows():
        lines.append(header)
        lines.append("| " + " | ".join(_clean_text(str(v)) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _markdown_repeat_header(md: str) -> str:
    """기존 마크다운 표 문자열에서 데이터 행 앞에 헤더를 반복해 반환."""
    if not md:
        return md

    lines = [line for line in md.splitlines() if line.strip()]
    if len(lines) <= 1:
        return md

    header = lines[0]
    sep_re = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")
    data_lines: list[str] = []
    for line in lines[1:]:
        if sep_re.match(line):
            continue
        data_lines.append(line)

    if not data_lines:
        return md

    out: list[str] = []
    for dl in data_lines:
        out.append(header)
        out.append(dl)
    return "\n".join(out)


def _extract_pdf_with_tables(pdf_path: Path) -> tuple[str, list[dict], Dict[int, str], int]:
    """
    PDF에서 본문 텍스트는 PyMuPDF로 추출하고,
    표는 PyMuPDF find_tables()와 Tabula로 추출해 마크다운 테이블로 반환한다.
    - 페이지별로 텍스트와 표를 분리 추출
    - 페이지 정보는 메타데이터에만 저장 (텍스트에는 페이지 마커 없음)
    - 이미지는 처리하지 않음
    - Tabula가 없거나 실패 시: PyMuPDF find_tables()로 폴백
    
    Returns:
        tuple: (전체 텍스트, 표 리스트, 페이지별 텍스트 딕셔너리, 총 페이지 수)
    """
    import fitz  # PyMuPDF
    
    page_texts: list[tuple[int, str]] = []
    all_tables: list[dict] = []
    total_pages = 0
    
    # with 컨텍스트 매니저로 안전하게 문서 열기/닫기
    try:
        with fitz.open(pdf_path) as doc:
            # 페이지 수를 먼저 저장 (doc이 닫히기 전에)
            total_pages = doc.page_count
            
            for page_idx in range(total_pages):
                page_num = page_idx + 1
                page = doc[page_idx]
                page_tables: list[dict] = []
                
                # 1. 일반 텍스트 추출 (PyMuPDF)
                try:
                    raw_text = page.get_text()
                    if raw_text and raw_text.strip():
                        clean_text = _clean_text(raw_text)
                        # 페이지 마커 없이 순수 텍스트만 저장 (페이지 정보는 메타데이터에 저장)
                        page_texts.append((page_num, clean_text))  # (페이지번호, 텍스트) 튜플로 저장
                except Exception as e:
                    logger.warning(f"[PyMuPDF] 텍스트 추출 실패 (p{page_num}): {e}")
                
                # 2. 표 추출 (PyMuPDF 내장 테이블 감지)
                try:
                    tables = page.find_tables()
                    table_list = getattr(tables, "tables", tables) if tables else []
                    
                    if table_list:
                        logger.info(f"[PyMuPDF] {page_num}페이지: 테이블 {len(table_list)}개 탐지")
                    
                    for i, table in enumerate(table_list):
                        try:
                            table_data = table.extract()
                            if not table_data:
                                logger.debug(f"[PyMuPDF] {page_num}페이지 테이블 {i+1}: 빈 데이터")
                                continue
                            
                            # 마크다운 테이블 생성
                            markdown_lines: list[str] = []
                            
                            # 헤더
                            header_row = [_clean_text(str(cell or "")) for cell in table_data[0]]
                            markdown_lines.append("| " + " | ".join(header_row) + " |")
                            markdown_lines.append("|" + "---|" * len(header_row))
                            
                            # 데이터 행
                            for row in table_data[1:]:
                                clean_row = [_clean_text(str(cell or "")) for cell in row]
                                markdown_lines.append("| " + " | ".join(clean_row) + " |")
                            
                            md = "\n".join(markdown_lines)
                            md = _markdown_repeat_header(md)  # 헤더 반복 적용
                            
                            if md.strip():
                                rect = fitz.Rect(*table.bbox) if hasattr(table, 'bbox') else None
                                page_tables.append({
                                    "page": page_num,
                                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)] if rect else [],
                                    "text": md,
                                })
                                logger.info(f"[PyMuPDF] {page_num}페이지 테이블 {i+1}: 마크다운 변환 성공, 길이={len(md)}")
                        except Exception as e:
                            logger.exception(f"[PyMuPDF] {page_num}페이지 테이블 {i+1} 처리 실패: {e}")
                except Exception as e:
                    logger.warning(f"[PyMuPDF] {page_num}페이지 표 감지 실패: {e}")
                
                # 3. Tabula로 추가 표 추출 (선택)
                try:
                    import tabula  # requires Java (JRE/JDK)
                    TABULA_AVAILABLE = True
                except ImportError:
                    TABULA_AVAILABLE = False
                
                if TABULA_AVAILABLE:
                    try:
                        dfs = tabula.read_pdf(
                            str(pdf_path),
                            pages=page_num,
                            multiple_tables=True,
                            lattice=True,
                            silent=True,
                        )
                        if not dfs:
                            dfs = tabula.read_pdf(
                                str(pdf_path),
                                pages=page_num,
                                multiple_tables=True,
                                stream=True,
                                silent=True,
                            )
                        
                        if not dfs:
                            logger.debug(f"[Tabula] {page_num}페이지: 탐지된 테이블 없음")
                        else:
                            logger.info(f"[Tabula] {page_num}페이지: 테이블 {len(dfs)}개 탐지, shapes={[df.shape for df in dfs]}")
                        
                        for j, df in enumerate(dfs or []):
                            try:
                                if df.empty:
                                    logger.debug(f"[Tabula] {page_num}페이지 테이블 {j+1}: empty DataFrame - 건너뜀")
                                    continue
                                
                                logger.info(f"[Tabula] {page_num}페이지 테이블 {j+1}: columns={list(df.columns)}, shape={df.shape}")
                                
                                md = _df_to_markdown_repeat_header(df)
                                md = _clean_text(md)
                                
                                if md:
                                    # PyMuPDF에서 이미 같은 페이지의 표를 찾았는지 확인 (중복 제거)
                                    is_duplicate = False
                                    for existing_table in page_tables:
                                        if existing_table.get("page") == page_num:
                                            # 간단한 중복 체크: 첫 줄이 비슷하면 중복으로 간주
                                            existing_first_line = existing_table.get("text", "").split("\n")[0] if existing_table.get("text") else ""
                                            new_first_line = md.split("\n")[0] if md else ""
                                            if existing_first_line and new_first_line and existing_first_line[:50] == new_first_line[:50]:
                                                is_duplicate = True
                                                logger.debug(f"[Tabula] {page_num}페이지 테이블 {j+1}: PyMuPDF와 중복으로 판단, 건너뜀")
                                                break
                                    
                                    if not is_duplicate:
                                        page_tables.append({
                                            "page": page_num,
                                            "bbox": [],
                                            "text": md,
                                        })
                                        logger.info(f"[Tabula] {page_num}페이지 테이블 {j+1}: 마크다운 변환 성공, 길이={len(md)}")
                                else:
                                    logger.debug(f"[Tabula] {page_num}페이지 테이블 {j+1}: 마크다운 변환 후 빈 문자열")
                            except Exception as e:
                                logger.exception(f"[Tabula] {page_num}페이지 테이블 {j+1} 처리 실패: {e}")
                                continue
                    except Exception as e:
                        logger.debug(f"[Tabula] {page_num}페이지 표 추출 실패: {e}")
                
                # 페이지별 표를 전체 리스트에 추가
                all_tables.extend(page_tables)
    
    except Exception:
        logger.exception(f"Failed to open PDF: {pdf_path}")
        return "", [], {}, 0
    
    # 모든 페이지 텍스트 결합 (페이지 마커 없이)
    pdf_text = _clean_text("\n\n".join(text for _, text in page_texts if text))
    
    # 페이지별 텍스트 딕셔너리 생성 (메타데이터용)
    pages_text_dict: Dict[int, str] = {}
    for page_num, text_content in page_texts:
        if text_content:
            pages_text_dict[page_num] = text_content

    # 표 추출 결과 로깅
    if all_tables:
        logger.info(f"[Extract] {pdf_path.name}: 총 {len(all_tables)}개 표 추출 완료")
    else:
        logger.info(f"[Extract] {pdf_path.name}: 추출된 표 없음")
    
    return pdf_text, all_tables, pages_text_dict, total_pages
