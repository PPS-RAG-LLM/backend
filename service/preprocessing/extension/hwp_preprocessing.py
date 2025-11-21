# """
# HWP 전처리 모듈
# HWP 파일을 텍스트와 표로 추출하는 기능 제공
# """
# from __future__ import annotations
# import logging
# from pathlib import Path
# from typing import Optional

# logger = logging.getLogger(__name__)


# def _clean_text(s: str | None) -> str:
#     """텍스트 정규화 (공통 유틸리티에서 import)"""
#     from service.preprocessing.extension.utils import _clean_text as _clean_text_util
#     return _clean_text_util(s)


# def _convert_via_libreoffice(src: Path, target_ext: str) -> Optional[Path]:
#     """LibreOffice를 통한 문서 변환"""
#     try:
#         import subprocess
#         import shutil
        
#         # LibreOffice 설치 확인
#         libreoffice_cmd = shutil.which("libreoffice")
#         if not libreoffice_cmd:
#             logger.warning(f"[LibreOffice] LibreOffice가 설치되지 않았습니다. 설치가 필요합니다.")
#             return None
        
#         outdir = src.parent
#         # 변환 실행 (stderr는 캡처하여 로깅)
#         result = subprocess.run(
#             [
#                 libreoffice_cmd, "--headless", "--convert-to", target_ext,
#                 "--outdir", str(outdir), str(src)
#             ],
#             check=False,  # check=False로 변경하여 에러를 직접 처리
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             timeout=120,  # 타임아웃 추가
#         )
        
#         if result.returncode != 0:
#             stderr_msg = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
#             logger.warning(
#                 f"[LibreOffice] 변환 실패 (returncode={result.returncode}): {src.name} -> {target_ext}. "
#                 f"에러: {stderr_msg[:200]}"
#             )
#             return None
        
#         # 변환된 파일 확인
#         cand = src.with_suffix("." + target_ext)
#         if cand.exists():
#             logger.debug(f"[LibreOffice] 변환 성공: {src.name} -> {cand.name}")
#             return cand
#         else:
#             logger.warning(f"[LibreOffice] 변환된 파일을 찾을 수 없습니다: {cand}")
#             return None
            
#     except subprocess.TimeoutExpired:
#         logger.error(f"[LibreOffice] 변환 타임아웃: {src.name} (120초 초과)")
#         return None
#     except FileNotFoundError:
#         logger.error(f"[LibreOffice] LibreOffice 명령어를 찾을 수 없습니다. 설치가 필요합니다.")
#         return None
#     except Exception as e:
#         logger.exception(f"[LibreOffice] 변환 중 예외 발생: {src.name}, 오류: {e}")
#         return None


# def _extract_hwp(fp: Path) -> tuple[str, list[dict]]:
#     """HWP 파일 추출
#     LibreOffice를 사용하여 PDF로 변환한 후 PDF 전처리 함수를 호출합니다.
#     - 임시 PDF 파일 생성 → PDF 전처리 → 임시 파일 삭제
#     - 반환: (본문텍스트, 표리스트[])
#     """
#     # LibreOffice를 사용하여 PDF로 변환
#     pdf_path = _convert_via_libreoffice(fp, "pdf")
    
#     if pdf_path and pdf_path.exists():
#         try:
#             # PDF 전처리 함수 호출
#             from service.preprocessing.extension.pdf_preprocessing import _extract_pdf_with_tables
            
#             # PDF 전처리 함수는 (text, tables, pages_text_dict, total_pages) 반환
#             # 여기서는 (text, tables)만 필요
#             text, tables, _, _ = _extract_pdf_with_tables(pdf_path)
            
#             logger.info(f"[HWP] PDF 변환 후 추출 성공: {fp.name} -> {len(text)}자, 표 {len(tables)}개")
#             return text, tables
#         except Exception as e:
#             logger.exception(f"[HWP] PDF 변환 후 추출 실패: {fp.name}, 오류: {e}")
#         finally:
#             # 임시 PDF 파일 삭제
#             try:
#                 if pdf_path.exists():
#                     pdf_path.unlink()
#                     logger.debug(f"[HWP] 임시 PDF 파일 삭제: {pdf_path}")
#             except Exception as e:
#                 logger.warning(f"[HWP] 임시 PDF 파일 삭제 실패: {pdf_path}, 오류: {e}")
    
#     # PDF 변환 실패 시 폴백: 기존 방법들 시도
#     logger.warning(f"[HWP] PDF 변환 실패, 폴백 방법 시도: {fp.name}")
    
#     # --- 폴백 1: hwp5txt CLI 시도 ---
#     def _try_hwp5txt_cli(path: Path) -> Optional[str]:
#         """hwp5txt CLI로 텍스트 추출"""
#         try:
#             import shutil, subprocess
#             if shutil.which("hwp5txt") is None:
#                 return None
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
    
#     cli_text = _try_hwp5txt_cli(fp)
#     if cli_text is not None:
#         logger.info(f"[HWP] hwp5txt CLI로 추출 성공: {fp.name}")
#         return cli_text, []
    
#     # --- 폴백 2: LibreOffice 변환(docx) 시도 ---
#     conv = _convert_via_libreoffice(fp, "docx")
#     if conv and conv.exists():
#         try:
#             from service.preprocessing.extension.docx_preprocessing import _extract_docx
#             text, tables = _extract_docx(conv)
#             if text or tables:
#                 logger.info(f"[HWP] DOCX 변환 후 추출 성공: {fp.name}")
#                 return text, tables
#         finally:
#             try:
#                 conv.unlink()
#             except Exception:
#                 pass
    
#     logger.warning(f"[HWP] 모든 추출 방법 실패: {fp.name}")
#     return "", []

