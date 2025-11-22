# """
# HWP 전처리 모듈
# HWP 파일을 텍스트와 표로 추출하는 기능 제공
# """
# from __future__ import annotations
# import logging
# from pathlib import Path
# from typing import List, Optional, Tuple

# logger = logging.getLogger(__name__)


# def _clean_text(s: str | None) -> str:
#     """텍스트 정규화 (pdf_preprocessing에서 import)"""
#     from service.preprocessing.extension.pdf_preprocessing import _clean_text as _clean_text_pdf
#     return _clean_text_pdf(s)


# def _convert_via_libreoffice(src: Path, target_ext: str) -> Optional[Path]:
#     """LibreOffice를 통한 문서 변환"""
#     try:
#         import subprocess
#         outdir = src.parent
#         subprocess.run([
#             "libreoffice", "--headless", "--convert-to", target_ext, 
#             "--outdir", str(outdir), str(src)
#         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         cand = src.with_suffix("." + target_ext)
#         return cand if cand.exists() else None
#     except Exception:
#         return None


# def _extract_hwp(fp: Path) -> Tuple[str, List[dict]]:
#     """HWP 파일 추출 (다단계 폴백)
#     순서: python-hwp(API) -> hwp5txt(CLI) -> LibreOffice 변환(docx) -> olefile(구버전 시도)
#     반환: (본문텍스트, 표리스트[])
#     """
#     # --- 0) 공통 헬퍼 ---
#     def _try_hwp5txt_cli(path: Path) -> Optional[str]:
#         """hwp5txt CLI로 텍스트 추출 (권장 경로)"""
#         try:
#             import shutil, subprocess
#             if shutil.which("hwp5txt") is None:
#                 return None
#             # hwp5txt 출력은 stdout. 기본 인코딩은 UTF-8로 가정
#             res = subprocess.run(
#                 ["hwp5txt", str(path)],
#                 check=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 timeout=60,
#             )
#             out = res.stdout.decode("utf-8", errors="ignore")
#             return _clean_text(out) if out else ""
#         except Exception:
#             return None

#     # --- 1) python-hwp 모듈 시도 ---
#     try:
#         import pyhwp
#         from pyhwp.hwp5.xmlmodel import Hwp5File
#         texts, tables = [], []
#         with Hwp5File(str(fp)) as hwp:
#             # 단순 본문 루프 (구조가 달라도 최대한 텍스트 회수)
#             for section in getattr(hwp.bodytext, "sections", []):
#                 # 문단
#                 for paragraph in getattr(section, "paragraphs", []):
#                     try:
#                         t = paragraph.get_text()
#                         if t and t.strip():
#                             texts.append(_clean_text(t))
#                     except Exception:
#                         continue
#                 # 표(가능하면)
#                 for table in getattr(section, "tables", []):
#                     try:
#                         rows = []
#                         for row in getattr(table, "rows", []):
#                             cells = []
#                             for cell in getattr(row, "cells", []):
#                                 ctext = cell.get_text() if hasattr(cell, "get_text") else str(cell)
#                                 cells.append(_clean_text(ctext))
#                             if cells:
#                                 rows.append(cells)
#                         if rows:
#                             md = "\n".join("| " + " | ".join(r) + " |" for r in rows)
#                             tables.append({"page": 0, "bbox": [], "text": md})
#                     except Exception:
#                         continue
#         text_joined = _clean_text("\n\n".join(texts))
#         if text_joined or tables:
#             return text_joined, tables
#     except Exception as e:
#         logger.debug(f"python-hwp extraction failed for {fp}: {e}")

#     # --- 2) hwp5txt CLI 시도 (가장 잘 되는 편) ---
#     cli_text = _try_hwp5txt_cli(fp)
#     if cli_text is not None:
#         return cli_text, []

#     # --- 3) LibreOffice 변환(docx) 시도 (대부분 실패하지만 폴백으로 유지) ---
#     conv = _convert_via_libreoffice(fp, "docx")
#     if conv and conv.exists():
#         try:
#             # docx 추출 함수 import (rag_preprocessing에서)
#             from service.preprocessing.rag_preprocessing import _extract_docx
#             text, tables = _extract_docx(conv)
#         finally:
#             try:
#                 conv.unlink()
#             except Exception:
#                 pass
#         if text or tables:
#             return text, tables

#     # --- 4) olefile (구버전 HWP에만 가끔) ---
#     try:
#         import olefile
#         if olefile.isOleFile(str(fp)):
#             with olefile.OleFileIO(str(fp)) as ole:
#                 text_content = ""
#                 for stream in ole.listdir():
#                     try:
#                         # 본문 후보 스트림 이름 휴리스틱
#                         sname = "/".join(stream)
#                         if any(k in sname.lower() for k in ("bodytext", "prvtext", "section", "paragraph")):
#                             data = ole.openstream(stream).read()
#                             text_content += data.decode("utf-8", errors="ignore")
#                     except Exception:
#                         continue
#                 if text_content.strip():
#                     return _clean_text(text_content), []
#     except Exception as e:
#         logger.debug(f"olefile extraction failed for {fp}: {e}")

#     logger.warning(f"HWP 파일 추출 실패: {fp}. python-hwp, hwp5txt, LibreOffice, olefile 모두 실패.")
#     return "", []

