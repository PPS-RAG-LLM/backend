# === Vector DB Service (Milvus Server, Pro) ===
# - 작업유형(task_type)별 보안레벨 관리: doc_gen | summary | qna
# - Milvus Docker 서버 전용 (Lite 제거)
# - 벡터/하이브리드 검색 지원, 실행 로그 적재

from __future__ import annotations
import uuid
import json
import os
import time
import logging
import re
import unicodedata

# sqlite3 제거
import shutil
import threading
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict, Counter

import torch
from pydantic import BaseModel, Field
from pymilvus import MilvusClient, DataType

try:
    # Milvus 2.4+ Function/BM25 하이브리드
    from pymilvus import Function, FunctionType
except Exception:
    Function = None
    class FunctionType:
        BM25 = "BM25"
from transformers import AutoModel, AutoTokenizer

# ORM 추가 임포트
from utils.database import get_session
from storage.db_models import (
    EmbeddingModel,
    RagSettings,
    SecurityLevelConfigTask,
    SecurityLevelKeywordsTask,
)
# 밀버스 DB 글자 수 제한 
MILVUS_TEXT_VARCHAR_MAX = 32768
MILVUS_TEXT_SAFE_LIMIT = 32000  # 여유 버퍼를 둔 안전 상한
def _split_text_for_milvus(s: str, limit: int = MILVUS_TEXT_SAFE_LIMIT) -> list[str]:
    """Milvus VARCHAR 안전 길이로 텍스트를 분할한다."""
    s = s or ""
    if len(s) <= limit:
        return [s]
    # 1차: 줄 단위로 누적 분할
    parts, buf, blen = [], [], 0
    for line in s.splitlines(True):  # 개행 보존
        if blen + len(line) > limit and buf:
            parts.append("".join(buf).strip())
            buf, blen = [line], len(line)
        else:
            buf.append(line); blen += len(line)
    if buf:
        parts.append("".join(buf).strip())

    # 2차: 여전히 넘치는 덩어리는 하드 슬라이스
    out = []
    for p in parts:
        if len(p) <= limit:
            if p: out.append(p)
        else:
            for i in range(0, len(p), limit):
                chunk = p[i:i+limit].strip()
                if chunk: out.append(chunk)
    return out

# KST 시간 포맷 유틸
from utils.time import now_kst, now_kst_string

logger = logging.getLogger(__name__)

# -------------------------------------------------
# 경로 상수
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
STORAGE_DIR = PROJECT_ROOT / "storage"
USER_DATA_ROOT = STORAGE_DIR / "user_data"
RAW_DATA_DIR = USER_DATA_ROOT / "row_data"
LOCAL_DATA_ROOT = USER_DATA_ROOT / "preprocessed_data"  # 유지(폴더 구조 호환)
RESOURCE_DIR = (BASE_DIR / "resources").resolve()
EXTRACTED_TEXT_DIR = (PROJECT_ROOT / "storage" / "extracted_texts").resolve()
META_JSON_PATH = EXTRACTED_TEXT_DIR / "_extraction_meta.json"
MODEL_ROOT_DIR = (PROJECT_ROOT / "storage" / "embedding-models").resolve()

SQLITE_DB_PATH = (PROJECT_ROOT / "storage" / "pps_rag.db").resolve()

VAL_SESSION_ROOT = (STORAGE_DIR / "val_data").resolve()
SESSIONS_INDEX_PATH = (VAL_SESSION_ROOT / "_sessions.json").resolve()

# Milvus Server 접속 정보 (환경변수로 오버라이드 가능)
MILVUS_URI = os.getenv("MILVUS_URI", "http://biz.ppsystem.co.kr:3006")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", None)  # 예: "root:Milvus" (인증 사용 시)
COLLECTION_NAME = "pdf_chunks_pro"

# 작업유형
TASK_TYPES = ("doc_gen", "summary", "qna")

# 지원 확장자
SUPPORTED_EXTS = {".pdf", ".txt", ".text", ".md", ".docx", ".pptx", ".csv", ".xlsx", ".xls", ".doc", ".ppt", ".hwp"}

# 텍스트 정리용 정규식
ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')
CONTROL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')  # \t,\n은 유지
MULTISPACE_LINE_END_RE = re.compile(r'[ \t]+\n')
NEWLINES_RE = re.compile(r'\n{3,}')

_CURRENT_EMBED_MODEL_KEY = "qwen3_0_6b"
_CURRENT_SEARCH_TYPE = "hybrid"
_CURRENT_CHUNK_SIZE = 512
_CURRENT_OVERLAP = 64


# -------------------------------------------------
# 텍스트 정리 및 다중 확장자 지원
# -------------------------------------------------

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


def _ext(p: Path) -> str:
    """파일 확장자 반환 (소문자)"""
    return p.suffix.lower()


def _extract_pdf_with_tables(pdf_path: Path) -> tuple[str, list[dict]]:
    """PDF에서 표와 본문을 분리 추출 (특수문자 정리 포함)"""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    page_texts: list[str] = []
    table_blocks_all: list[dict] = []
    
    for page_idx, page in enumerate(doc, start=1):
        # 표 찾기
        try:
            tf = page.find_tables()
            tables = getattr(tf, "tables", []) if tf else []
        except Exception:
            tables = []
        
        table_rects = []
        for t in tables:
            rect = fitz.Rect(*t.bbox)
            table_rects.append(rect)
            try:
                md = t.to_markdown()
            except Exception:
                try:
                    rows = t.extract()
                    md = "\n".join("| " + " | ".join(_clean_text(c or "")) + " |" for c in rows)
                except Exception:
                    md = ""
            md = _clean_text(md)
            if md:
                table_blocks_all.append({
                    "page": int(page_idx),
                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "text": md,
                })
        
        # 본문에서 표 영역 제외
        parts = []
        for x0, y0, x1, y1, btxt, *_ in page.get_text("blocks"):
            rect = fitz.Rect(x0, y0, x1, y1)
            if any(rect.intersects(r) for r in table_rects):
                continue
            if btxt and btxt.strip():
                parts.append(_clean_text(btxt))
        page_texts.append("\n".join(p for p in parts if p))
    
    pdf_text = _clean_text("\n\n".join(p for p in page_texts if p))
    return pdf_text, table_blocks_all


def _extract_plain_text(fp: Path) -> tuple[str, list[dict]]:
    """TXT/MD 파일 추출"""
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = fp.read_text(errors="ignore")
    return _clean_text(text), []


def _extract_docx(fp: Path) -> tuple[str, list[dict]]:
    """DOCX 파일 추출"""
    try:
        from docx import Document
    except Exception:
        logger.warning(f"python-docx not available, treating {fp} as plain text")
        return _extract_plain_text(fp)
    
    try:
        d = Document(str(fp))
        paras = [_clean_text(p.text) for p in d.paragraphs if _clean_text(p.text)]
        tables = []
        for tb in d.tables:
            rows = []
            for r in tb.rows:
                rows.append([_clean_text(c.text) for c in r.cells])
            if rows:
                md = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                tables.append({"page": 0, "bbox": [], "text": md})
        return _clean_text("\n\n".join(paras)), tables
    except Exception as e:
        logger.exception(f"Failed to extract DOCX {fp}: {e}")
        return _extract_plain_text(fp)


def _extract_pptx(fp: Path) -> tuple[str, list[dict]]:
    """PPTX 파일 추출"""
    try:
        from pptx import Presentation
    except Exception:
        logger.warning(f"python-pptx not available, skipping {fp}")
        return "", []
    
    try:
        prs = Presentation(str(fp))
        texts = []
        tables = []
        for i, slide in enumerate(prs.slides, start=1):
            for sh in slide.shapes:
                if hasattr(sh, "has_text_frame") and sh.has_text_frame:
                    txt = "\n".join(p.text for p in sh.text_frame.paragraphs if p.text)
                    txt = _clean_text(txt)
                    if txt:
                        texts.append(txt)
                if hasattr(sh, "has_table") and sh.has_table:
                    rows = []
                    for r in sh.table.rows:
                        rows.append([_clean_text(c.text) for c in r.cells])
                    if rows:
                        md = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                        tables.append({"page": i, "bbox": [], "text": md})
        return _clean_text("\n\n".join(texts)), tables
    except Exception as e:
        logger.exception(f"Failed to extract PPTX {fp}: {e}")
        return "", []


def _df_to_markdown(df, max_rows=500) -> str:
    """DataFrame을 마크다운 테이블로 변환"""
    if len(df) > max_rows:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_clean_text(str(v)) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _extract_csv(fp: Path) -> tuple[str, list[dict]]:
    """CSV 파일 추출"""
    try:
        import pandas as pd
        df = pd.read_csv(fp)
        md = _df_to_markdown(df)
        return "", [{"page": 0, "bbox": [], "text": md}]
    except Exception as e:
        logger.warning(f"Failed to extract CSV {fp}: {e}, trying as plain text")
        return _extract_plain_text(fp)


def _extract_excel(fp: Path) -> tuple[str, list[dict]]:
    """Excel 파일 추출"""
    try:
        import pandas as pd
        xls = pd.ExcelFile(fp)
        tables = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            md = f"### {name}\n" + _df_to_markdown(df)
            tables.append({"page": 0, "bbox": [], "text": md})
        return "", tables
    except Exception as e:
        logger.warning(f"Failed to extract Excel {fp}: {e}")
        return "", []


def _convert_via_libreoffice(src: Path, target_ext: str) -> Optional[Path]:
    """LibreOffice를 통한 문서 변환"""
    try:
        import subprocess
        outdir = src.parent
        subprocess.run([
            "libreoffice", "--headless", "--convert-to", target_ext, 
            "--outdir", str(outdir), str(src)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cand = src.with_suffix("." + target_ext)
        return cand if cand.exists() else None
    except Exception:
        return None


def _extract_doc(fp: Path) -> tuple[str, list[dict]]:
    """DOC 파일 추출 (LibreOffice 변환 시도)"""
    conv = _convert_via_libreoffice(fp, "docx")
    if conv and conv.exists():
        result = _extract_docx(conv)
        try:
            conv.unlink()  # 임시 변환 파일 삭제
        except Exception:
            pass
        return result
    return _extract_plain_text(fp)


def _extract_ppt(fp: Path) -> tuple[str, list[dict]]:
    """PPT 파일 추출 (LibreOffice 변환 시도)"""
    conv = _convert_via_libreoffice(fp, "pptx")
    if conv and conv.exists():
        result = _extract_pptx(conv)
        try:
            conv.unlink()  # 임시 변환 파일 삭제
        except Exception:
            pass
        return result
    return "", []


def _extract_hwp(fp: Path) -> tuple[str, list[dict]]:
    """HWP 파일 추출"""
    # 방법 1: python-hwp 라이브러리 시도
    try:
        import pyhwp
        from pyhwp.hwp5.xmlmodel import Hwp5File
        
        with Hwp5File(str(fp)) as hwp:
            texts = []
            tables = []
            
            # HWP 파일에서 텍스트 추출
            for section in hwp.bodytext.sections:
                for paragraph in section.paragraphs:
                    text = paragraph.get_text()
                    if text and text.strip():
                        texts.append(_clean_text(text))
                
                # 표 추출 시도
                for table in section.tables:
                    try:
                        rows = []
                        for row in table.rows:
                            cells = []
                            for cell in row.cells:
                                cell_text = cell.get_text() if hasattr(cell, 'get_text') else str(cell)
                                cells.append(_clean_text(cell_text))
                            if cells:
                                rows.append(cells)
                        
                        if rows:
                            md = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                            tables.append({"page": 0, "bbox": [], "text": md})
                    except Exception:
                        continue
            
            return _clean_text("\n\n".join(texts)), tables
    
    except Exception as e:
        logger.debug(f"python-hwp extraction failed for {fp}: {e}")
    
    # 방법 2: LibreOffice 변환 시도
    conv = _convert_via_libreoffice(fp, "docx")
    if conv and conv.exists():
        result = _extract_docx(conv)
        try:
            conv.unlink()  # 임시 변환 파일 삭제
        except Exception:
            pass
        return result
    
    # 방법 3: olefile을 사용한 기본 텍스트 추출 시도
    try:
        import olefile
        
        if olefile.isOleFile(str(fp)):
            with olefile.OleFileIO(str(fp)) as ole:
                # HWP 파일의 기본 텍스트 스트림 시도
                try:
                    # HWP 파일 구조에서 텍스트 추출 (간단한 방법)
                    streams = ole.listdir()
                    text_content = ""
                    
                    for stream in streams:
                        try:
                            if 'BodyText' in str(stream) or 'PrvText' in str(stream):
                                data = ole._olestream_size.get(stream, b'')
                                if data:
                                    # 간단한 텍스트 추출 (완전하지 않을 수 있음)
                                    text_content += data.decode('utf-8', errors='ignore')
                        except Exception:
                            continue
                    
                    if text_content.strip():
                        return _clean_text(text_content), []
                except Exception:
                    pass
    
    except Exception as e:
        logger.debug(f"olefile extraction failed for {fp}: {e}")
    
    # 모든 방법 실패 시 경고 로그 후 빈 결과 반환
    logger.warning(f"HWP 파일 추출 실패: {fp}. python-hwp, LibreOffice, olefile 모두 사용할 수 없습니다.")
    return "", []


def _extract_any(path: Path) -> tuple[str, list[dict]]:
    """통합 문서 추출 라우터"""
    ext = _ext(path)
    if ext == ".pdf":
        return _extract_pdf_with_tables(path)
    if ext in {".txt", ".text", ".md"}:
        return _extract_plain_text(path)
    if ext == ".docx":
        return _extract_docx(path)
    if ext == ".pptx":
        return _extract_pptx(path)
    if ext == ".csv":
        return _extract_csv(path)
    if ext in {".xlsx", ".xls"}:
        return _extract_excel(path)
    if ext == ".doc":
        return _extract_doc(path)
    if ext == ".ppt":
        return _extract_ppt(path)
    if ext == ".hwp":
        return _extract_hwp(path)
    # 모르는 확장자는 텍스트로 시도
    return _extract_plain_text(path)


# -------------------------------------------------
# 인제스트 파라미터 설정
# -------------------------------------------------
def set_ingest_params(chunk_size: int | None = None, overlap: int | None = None):
    # rag_settings 단일 소스로 저장
    set_vector_settings(chunk_size=chunk_size, overlap=overlap)


def get_ingest_params():
    row = _get_vector_settings_row()
    return {"chunkSize": row["chunk_size"], "overlap": row["overlap"]}


# -------------------------------------------------
# Pydantic 스키마
# -------------------------------------------------
class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, gt=0)
    user_level: int = Field(1, ge=1)
    task_type: str = Field(..., description="doc_gen | summary | qna")
    model: Optional[str] = None  # 내부적으로 settings에서 로드


class SinglePDFIngestRequest(BaseModel):
    pdf_path: str
    task_types: Optional[List[str]] = None  # 기본은 모든 작업유형
    workspace_id: Optional[int] = None


# -------------------------------------------------
# SQLite 유틸
# -------------------------------------------------


# ====== New helpers ======
def save_raw_file(filename: str, content: bytes) -> str:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DATA_DIR / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(content)
    return str(out)


def save_raw_to_row_data(f):
    """Save FastAPI UploadFile to row_data and return relative path."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(getattr(f, "filename", "uploaded"))
    dst = RAW_DATA_DIR / name.name
    if dst.exists():
        stem, ext = name.stem, name.suffix
        dst = RAW_DATA_DIR / f"{stem}_{int(time.time())}{ext}"
    with dst.open("wb") as out:
        data = f.file.read() if hasattr(f, "file") else b""
        out.write(data)
    try:
        return str(dst.relative_to(RAW_DATA_DIR))
    except Exception:
        return dst.name


# === Embedding cache(singleton) ===
_EMBED_CACHE: dict[str, tuple[any, any, any]] = {}  # key -> (tok, model, device)
_EMBED_ACTIVE_KEY: Optional[str] = None
_EMBED_LOCK = threading.Lock()


def _invalidate_embedder_cache():
    global _EMBED_CACHE, _EMBED_ACTIVE_KEY
    with _EMBED_LOCK:
        _EMBED_CACHE.clear()
        _EMBED_ACTIVE_KEY = None


def _get_or_load_embedder(model_key: str, preload: bool = False):
    """
    전역 캐시에서 (tok, model, device) 반환.
    - 캐시에 없으면 로드해서 저장(지연 로딩)
    - preload=True는 의미상 웜업 호출일 뿐, 반환 동작은 동일
    """
    global _EMBED_CACHE, _EMBED_ACTIVE_KEY
    if not model_key:
        raise ValueError(
            "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
        )

    with _EMBED_LOCK:
        if _EMBED_ACTIVE_KEY == model_key and model_key in _EMBED_CACHE:
            return _EMBED_CACHE[model_key]
        # 키가 바뀌면 캐시 전체 무효화(동시 2개 방지)
        _EMBED_CACHE.clear()
        tok, model, device = _load_embedder(model_key)
        _EMBED_CACHE[model_key] = (tok, model, device)
        _EMBED_ACTIVE_KEY = model_key
        return _EMBED_CACHE[model_key]


def warmup_active_embedder(logger_func=print):
    """
    서버 기동 시 호출용(선택). 활성 모델 키를 조회해 캐시를 채움.
    실패해도 서비스는 실제 사용 시 지연 로딩으로 복구됨.
    """
    try:
        key = _get_active_embedding_model_name()
        logger_func(f"[warmup] 활성 임베딩 모델: {key}. 로딩 시도...")
        _get_or_load_embedder(key, preload=True)
        logger_func(f"[warmup] 로딩 완료: {key}")
    except Exception as e:
        logger_func(f"[warmup] 로딩 실패(지연 로딩으로 복구 예정): {e}")


async def _get_or_load_embedder_async(model_key: str, preload: bool = False):
    """
    비동기 래퍼: blocking 함수(_get_or_load_embedder)를 스레드풀에서 실행
    이벤트 루프 블로킹 방지
    """
    loop = asyncio.get_running_loop()
    # blocking 함수(_get_or_load_embedder)를 스레드풀에서 실행
    return await loop.run_in_executor(None, _get_or_load_embedder, model_key, preload)


def _get_active_embedding_model_name() -> str:
    """활성화된 임베딩 모델 이름 반환 (없으면 예외)"""
    with get_session() as session:
        row = (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.is_active == 1)
            .order_by(EmbeddingModel.activated_at.desc().nullslast())
            .first()
        )
        if not row:
            raise ValueError(
                "활성화된 임베딩 모델이 없습니다. 먼저 /v1/admin/vector/settings에서 모델을 설정하세요."
            )
        return str(row.name)


def _set_active_embedding_model(name: str):
    with get_session() as session:
        # 존재하지 않으면 생성
        model = (
            session.query(EmbeddingModel).filter(EmbeddingModel.name == name).first()
        )
        if not model:
            model = EmbeddingModel(name=name, is_active=0)
            session.add(model)
            session.flush()
        # 모두 비활성 → 대상만 활성
        session.query(EmbeddingModel).filter(EmbeddingModel.is_active == 1).update(
            {"is_active": 0, "activated_at": None}
        )
        model.is_active = 1
        model.activated_at = now_kst()
        session.commit()


def _get_vector_settings_row() -> dict:
    """레거시 호환: rag_settings(싱글톤)에서 기본 청크 설정을 읽어온다."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {"search_type": "hybrid", "chunk_size": 512, "overlap": 64}
        return {
            "search_type": str(row.search_type or "hybrid"),
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }


def _get_rag_settings_row() -> dict:
    """RAG 전역 설정 로더. 없으면 빈 dict."""
    with get_session() as session:
        row = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not row:
            return {}
        return {
            "embedding_key": row.embedding_key,
            "search_type": row.search_type,
            "chunk_size": int(row.chunk_size or 512),
            "overlap": int(row.overlap or 64),
        }


def _update_vector_settings(
    search_type: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
):
    """레거시 API 호환: rag_settings(싱글톤) 업데이트"""
    cur = _get_vector_settings_row()
    new_search = (search_type or cur["search_type"]).lower()
    if new_search == "vector":
        new_search = "semantic"
    if new_search not in {"hybrid", "semantic", "bm25"}:
        raise ValueError(
            "unsupported searchType; allowed: 'hybrid','semantic','bm25' (or 'vector' alias)"
        )
    new_chunk = int(chunk_size if chunk_size is not None else cur["chunk_size"])
    new_overlap = int(overlap if overlap is not None else cur["overlap"])
    if new_chunk <= 0 or new_overlap < 0 or new_overlap >= new_chunk:
        raise ValueError("invalid chunk/overlap (chunk>0, 0 <= overlap < chunk)")

    with get_session() as session:
        s = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not s:
            s = RagSettings(id=1)
            session.add(s)
        s.search_type = new_search
        s.chunk_size = new_chunk
        s.overlap = new_overlap
        s.updated_at = now_kst()
        session.commit()


def _milvus_has_data(collection_name: str = COLLECTION_NAME) -> bool:
    client = _client()
    if collection_name not in client.list_collections():
        return False
    try:
        rows = client.query(collection_name=collection_name, output_fields=["pk"], limit=1)
        return len(rows) > 0
    except Exception:
        return True


# ---------------- Vector Settings ----------------
def set_vector_settings(embed_model_key: Optional[str] = None,
                        search_type: Optional[str] = None,
                        chunk_size: Optional[int] = None,
                        overlap: Optional[int] = None) -> Dict:
    """
    rag_settings 단일 소스로 설정 저장.
    - 임베딩 모델 변경 시 기존 데이터 존재하면 차단, 활성 모델 갱신 및 캐시 무효화
    - search_type/청크/오버랩은 rag_settings에만 반영
    """
    cur = get_vector_settings()
    key_now = cur.get("embeddingModel")
    st_now = (cur.get("searchType") or "hybrid").lower()
    cs_now = int(cur.get("chunkSize") or 512)
    ov_now = int(cur.get("overlap") or 64)

    new_key = embed_model_key or key_now
    new_st = (search_type or st_now).lower()
    # DB 제약과 일치(semantic == vector)
    if new_st == "semantic":
        new_st = "vector"
    if new_st not in {"hybrid", "bm25", "vector"}:
        raise ValueError("unsupported searchType; allowed: 'hybrid','bm25','vector'")

    new_cs = int(chunk_size if chunk_size is not None else cs_now)
    new_ov = int(overlap if overlap is not None else ov_now)
    if new_cs <= 0 or new_ov < 0 or new_ov >= new_cs:
        raise ValueError("invalid chunk/overlap (chunk>0, 0 <= overlap < chunk)")

    if embed_model_key is not None:
        if _milvus_has_data():
            raise RuntimeError("Milvus 컬렉션에 기존 데이터가 남아있습니다. 먼저 /v1/admin/vector/delete-all 을 호출해 초기화하세요.")
        _set_active_embedding_model(embed_model_key)
        _invalidate_embedder_cache()

    with get_session() as session:
        s = session.query(RagSettings).filter(RagSettings.id == 1).first()
        if not s:
            s = RagSettings(id=1)
            session.add(s)
        s.embedding_key = new_key
        # search_type/chunk/overlap은 _update_vector_settings에서 반영됨. 여기선 존재 시 보존
        if search_type is not None:
            s.search_type = (
                (search_type or "hybrid").lower().replace("vector", "semantic")
            )
        if chunk_size is not None:
            s.chunk_size = int(chunk_size)
        if overlap is not None:
            s.overlap = int(overlap)
        s.updated_at = now_kst()
        session.commit()

    return get_vector_settings()


def get_vector_settings() -> Dict:
    # rag_settings 는 검색 타입/청크/오버랩만 신뢰
    row = _get_vector_settings_row()  # {"search_type": "...", "chunk_size": 512, "overlap": 64}
    try:
        # ★활성 모델만 신뢰 (EmbeddingModel.is_active == 1)
        model = _get_active_embedding_model_name()
    except Exception:
        model = None

    return {
        "embeddingModel": model,                        # ← rag_settings.embedding_key는 무시
        "searchType": row.get("search_type", "hybrid"),
        "chunkSize": int(row.get("chunk_size", 512)),
        "overlap": int(row.get("overlap", 64)),
    }

# ------------- Security Level (per task) ---------
def _parse_at_string_to_keywords(value: str) -> List[str]:
    if not value:
        return []
    toks = [t.strip() for t in value.split("@")]
    return [t for t in toks if t]


def _normalize_keywords(val: Any) -> List[str]:
    """
    리스트/튜플/셋: 각 원소를 str로 캐스팅, 공백/해시 제거
    문자열: '@' 기준으로 토큰화
    빈 값 제거 및 중복 제거
    """
    out: List[str] = []
    if isinstance(val, str):
        toks = [t.strip() for t in val.split("@")]
    elif isinstance(val, (list, tuple, set)):
        toks = [str(t).strip() for t in val]
    else:
        toks = []
    for t in toks:
        if not t:
            continue
        if t.startswith("#"):
            t = t[1:]
        if t and t not in out:
            out.append(t)
    return out


def _normalize_levels(
    levels_raw: Dict[str, Any], max_level: int
) -> Dict[int, List[str]]:
    norm: Dict[int, List[str]] = {}
    for k, v in (levels_raw or {}).items():
        try:
            lv = int(str(k).strip().replace("level_", ""))
        except Exception:
            continue
        if lv < 1 or lv > max_level:
            continue
        kws = _normalize_keywords(v)
        if kws:
            norm[lv] = kws
    return norm


def upsert_security_level_for_task(
    task_type: str, max_level: int, levels_raw: Dict[str, Any]
) -> Dict:
    if task_type not in TASK_TYPES:
        raise ValueError(f"invalid task_type: {task_type}")
    if max_level < 1:
        raise ValueError("maxLevel must be >= 1")

    levels_map = _normalize_levels(levels_raw, max_level)

    with get_session() as session:
        # upsert config
        cfg = (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )
        if not cfg:
            cfg = SecurityLevelConfigTask(task_type=task_type, max_level=int(max_level))
            session.add(cfg)
        else:
            cfg.max_level = int(max_level)
            cfg.updated_at = now_kst()
        # replace keywords
        session.query(SecurityLevelKeywordsTask).filter(
            SecurityLevelKeywordsTask.task_type == task_type
        ).delete()
        for lv, kws in levels_map.items():
            for kw in kws:
                session.add(
                    SecurityLevelKeywordsTask(
                        task_type=task_type, level=int(lv), keyword=str(kw)
                    )
                )
        session.commit()
        return get_security_level_rules_for_task(task_type)


def get_security_level_rules_for_task(task_type: str) -> Dict:
    with get_session() as session:
        cfg = (
            session.query(SecurityLevelConfigTask)
            .filter(SecurityLevelConfigTask.task_type == task_type)
            .first()
        )
        max_level = int(cfg.max_level) if cfg else 1
        res: Dict[str, Any] = {
            "taskType": task_type,
            "maxLevel": max_level,
            "levels": {str(i): [] for i in range(1, max_level + 1)},
        }
        rows = (
            session.query(
                SecurityLevelKeywordsTask.level, SecurityLevelKeywordsTask.keyword
            )
            .filter(SecurityLevelKeywordsTask.task_type == task_type)
            .order_by(
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        for lv, kw in rows:
            key = str(int(lv))
            res["levels"].setdefault(key, []).append(str(kw))
        return res


def set_security_level_rules_per_task(config: Dict[str, Dict]) -> Dict:
    """
    config = {
      "doc_gen": {"maxLevel": 3, "levels": {"2": "@금액@연봉", "3": "@부정@퇴직금"}},
      "summary": {"maxLevel": 2, "levels": {"2": "@사내비밀"}},
      "qna": {"maxLevel": 3, "levels": {"2": "@연구", "3": "@개인정보"}}
    }
    """
    with get_session() as session:
        # 전체 삭제 후 재삽입(간결/명확)
        session.query(SecurityLevelConfigTask).delete()
        session.query(SecurityLevelKeywordsTask).delete()
        session.flush()

        for task in TASK_TYPES:
            entry = config.get(task) or {}
            max_level = int(entry.get("maxLevel", 1))
            session.add(
                SecurityLevelConfigTask(task_type=task, max_level=max(1, max_level))
            )
            levels = entry.get("levels", {}) or {}
            for lvl_str, at_str in levels.items():
                try:
                    lvl = int(str(lvl_str).strip().replace("level_", ""))
                except Exception:
                    continue
                if lvl <= 1 or lvl > max_level:
                    continue
                for kw in _parse_at_string_to_keywords(str(at_str)):
                    session.add(
                        SecurityLevelKeywordsTask(
                            task_type=task, level=int(lvl), keyword=str(kw)
                        )
                    )
        session.commit()
        return get_security_level_rules_all()


def get_security_level_rules_all() -> Dict:
    with get_session() as session:
        # 기본 max_level=1
        max_map = {t: 1 for t in TASK_TYPES}
        for task, max_level in session.query(
            SecurityLevelConfigTask.task_type, SecurityLevelConfigTask.max_level
        ).all():
            max_map[task] = int(max_level)

        res: Dict[str, Dict] = {}
        for task in TASK_TYPES:
            res[task] = {
                "maxLevel": max_map.get(task, 1),
                "levels": {str(i): [] for i in range(1, max_map.get(task, 1) + 1)},
            }

        rows = (
            session.query(
                SecurityLevelKeywordsTask.task_type,
                SecurityLevelKeywordsTask.level,
                SecurityLevelKeywordsTask.keyword,
            )
            .order_by(
                SecurityLevelKeywordsTask.task_type.asc(),
                SecurityLevelKeywordsTask.level.asc(),
                SecurityLevelKeywordsTask.keyword.asc(),
            )
            .all()
        )
        for task, level, kw in rows:
            if task in res:
                lv = str(int(level))
                if lv not in res[task]["levels"]:
                    res[task]["levels"][lv] = []
                res[task]["levels"][lv].append(str(kw))
        return res


def _determine_level_for_task(text: str, task_rules: Dict) -> int:
    max_level = int(task_rules.get("maxLevel", 1))
    levels = task_rules.get("levels", {})
    sel = 1
    # 상위 레벨 우선
    for lvl in range(1, max_level + 1):
        kws = levels.get(str(lvl), [])
        for kw in kws:
            if kw and kw in text:
                sel = max(sel, lvl)
    return sel


# -------------------------------------------------
# 모델 로딩/임베딩
# -------------------------------------------------
# --- replace this function definition entirely ---
def _resolve_model_input(model_key: Optional[str]) -> Tuple[str, Path]:
    """
    모델 키(=embedding_models.name)를 받아서 실제 로컬 디렉토리 Path를 결정한다.
    - DB(embedding_models)에서 is_active=1 AND name=model_key 인 행의 model_path가 유효하면 그것을 최우선 사용
    - 아니면 기존 폴더 스캔 로직(./storage/embedding-models/*)으로 fallback
    """
    key = (model_key or "bge").lower()

    # 1) DB에서 활성 모델의 model_path 우선 사용
    try:
        with get_session() as session:
            from storage.db_models import EmbeddingModel  # 안전 import
            row = (
                session.query(EmbeddingModel)
                .filter(EmbeddingModel.is_active == 1, EmbeddingModel.name == model_key)
                .order_by(EmbeddingModel.activated_at.desc().nullslast())
                .first()
            )
            if row and row.model_path:
                mp = Path(row.model_path).resolve()
                if mp.exists() and mp.is_dir():
                    return str(row.name), mp
    except Exception:
        logger.exception("[Embedding Model] DB lookup for active model_path failed")

    # 2) 기존 폴더 스캔 fallback
    cands: List[Path] = []
    if MODEL_ROOT_DIR.exists():
        for p in MODEL_ROOT_DIR.iterdir():
            if p.is_dir():
                cands.append(p.resolve())

    def aliases(p: Path) -> List[str]:
        nm = p.name.lower()
        res = [nm]
        if nm.startswith("embedding_"):
            res.append(nm[len("embedding_") :])
        return res

    for p in cands:
        if key in aliases(p):
            return p.name, p
    for p in cands:
        if key in p.name.lower():
            return p.name, p

    # fallback: qwen3_0_6b
    for p in cands:
        if "qwen3_0_6b" in p.name.lower():
            return p.name, p
    fb = MODEL_ROOT_DIR / "qwen3_0_6b"
    return fb.name, fb


# --- add: test 컬렉션 전용 보조 함수 ---
def _ensure_collection_and_index_for(
    client: MilvusClient,
    collection_name: str,
    emb_dim: int,
    metric: str = "IP",
):
    """
    _ensure_collection_and_index의 'collection_name' 파라미터 버전 (세션 컬렉션용)
    """
    logger.info(f"[Milvus] 컬렉션/인덱스 준비: {collection_name}")

    cols = client.list_collections()
    if collection_name not in cols:
        logger.info(f"[Milvus] 컬렉션 생성: {collection_name}")
        schema = client.create_schema(
            auto_id=True, enable_dynamic_field=False, description=f"PDF chunks ({collection_name})"
        )
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
        schema.add_field("path", DataType.VARCHAR, max_length=500)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field("task_type", DataType.VARCHAR, max_length=16)  # doc_gen|summary|qna
        schema.add_field("security_level", DataType.INT64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("version", DataType.INT64)

        # text / text_sparse (BM25)
        try:
            schema.add_field("text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
        except TypeError:
            schema.add_field("text", DataType.VARCHAR, max_length=32768)
        try:
            schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        except Exception:
            logger.warning("[Milvus] SPARSE_FLOAT_VECTOR 미지원 클라이언트 - 서버 BM25 하이브리드 불가")

        # BM25 Function
        if Function is not None:
            try:
                fn = Function(
                    name="bm25_text2sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(fn)
                logger.info("[Milvus] BM25 Function 연결 완료 (text -> text_sparse)")
            except Exception as e:
                logger.warning(f"[Milvus] BM25 Function 추가 실패: {e}")

        client.create_collection(collection_name=collection_name, schema=schema)
        logger.info(f"[Milvus] 컬렉션 생성 완료: {collection_name}")

    # 1) dense index
    try:
        idx_dense = client.list_indexes(collection_name=collection_name, field_name="embedding")
    except Exception:
        idx_dense = []
    if not idx_dense:
        ip = client.prepare_index_params()
        ip.add_index("embedding", "FLAT", metric_type=metric, params={})
        client.create_index(collection_name, ip, timeout=180.0, sync=True)

    # 2) sparse index
    try:
        idx_sparse = client.list_indexes(collection_name=collection_name, field_name="text_sparse")
    except Exception:
        idx_sparse = []
    if not idx_sparse:
        ip2 = client.prepare_index_params()
        try:
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", params={})
        except TypeError:
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
        client.create_index(collection_name, ip2, timeout=180.0, sync=True)

    # reload
    try:
        client.release_collection(collection_name=collection_name)
    except Exception:
        pass
    client.load_collection(collection_name=collection_name)
    logger.info(f"[Milvus] 로드 완료: {collection_name}")


def _load_embedder(model_key: Optional[str]) -> Tuple[any, any, any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, model_dir = _resolve_model_input(model_key)
    need_files = [
        model_dir / "tokenizer_config.json",
        model_dir / "tokenizer.json",
        model_dir / "config.json",
    ]

    # 모델 파일 누락 빠른 실패
    missing_files = [f for f in need_files if not f.exists()]
    if missing_files:
        logger.error(f"[Embedding Model] 필수 파일 누락: {model_dir}")
        logger.error(
            f"[Embedding Model] 누락된 파일들: {[str(f) for f in missing_files]}"
        )
        raise FileNotFoundError(f"[Embedding Model] 필수 파일 누락: {model_dir}")

    logger.info(f"[Embedding Model] 모델 로딩 시작: {model_key} from {model_dir}")
    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        .to(device)
        .eval()
    )
    logger.info(f"[Embedding Model] 모델 로딩 완료: {model_key}")
    return tok, model, device


def _mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def _embed_text(tok, model, device, text: str, max_len: int = 512):
    inputs = tok(
        text,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outs = model(**inputs)
    vec = (
        _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
    )
    return vec


# -------------------------------------------------
# Milvus Client / 컬렉션 스키마
# -------------------------------------------------
def _client() -> MilvusClient:
    kwargs = {"uri": MILVUS_URI}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)

def _ensure_collection_and_index(client: MilvusClient, emb_dim: int, metric: str = "IP", collection_name: str = COLLECTION_NAME):
    logger.info(f"[Milvus] 컬렉션 및 인덱스 준비 시작: {COLLECTION_NAME}")
    cols = client.list_collections()
    if COLLECTION_NAME not in cols:
        logger.info(f"[Milvus] 컬렉션 생성: {COLLECTION_NAME}")
        schema = client.create_schema(
            auto_id=True, enable_dynamic_field=False, description="PDF chunks (pro)"
        )
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=int(emb_dim))
        schema.add_field("path", DataType.VARCHAR, max_length=500)
        schema.add_field("chunk_idx", DataType.INT64)
        schema.add_field(
            "task_type", DataType.VARCHAR, max_length=16
        )  # 'doc_gen'|'summary'|'qna'
        schema.add_field("security_level", DataType.INT64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("version", DataType.INT64)
        # 하이브리드용 텍스트/스파스 필드
        # text: 본문 청크(분석기 활성), text_sparse: BM25 스파스 벡터
        try:
            schema.add_field("text", DataType.VARCHAR, max_length=32768, enable_analyzer=True)
        except TypeError:
            schema.add_field("text", DataType.VARCHAR, max_length=32768)
        try:
            schema.add_field("text_sparse", DataType.SPARSE_FLOAT_VECTOR)
        except Exception:
            logger.warning("[Milvus] SPARSE_FLOAT_VECTOR 미지원 클라이언트입니다. 서버 BM25 하이브리드 사용 불가.")

        # BM25 함수(가능한 경우: text -> text_sparse 자동생성)
        if Function is not None:
            try:
                fn = Function(
                    name="bm25_text2sparse",
                    function_type=FunctionType.BM25,
                    input_field_names=["text"],
                    output_field_names=["text_sparse"],
                )
                schema.add_function(fn)
                logger.info("[Milvus] BM25 Function 연결 완료 (text -> text_sparse)")
            except Exception as e:
                logger.warning(f"[Milvus] BM25 Function 추가 실패: {e}")
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
        logger.info(f"[Milvus] 컬렉션 생성 완료: {COLLECTION_NAME}")

    # 1) 덴스 벡터 인덱스(embedding)
    try:
        idx_dense = client.list_indexes(collection_name=COLLECTION_NAME, field_name="embedding")
    except Exception:
        idx_dense = []
    if not idx_dense:
        logger.info(f"[Milvus] (embedding) 인덱스 생성 시작")
        ip = client.prepare_index_params()
        # 덴스 벡터: 기본은 FLAT 유지(환경에 따라 HNSW/IVF 등으로 변경 가능)
        ip.add_index("embedding", "FLAT", metric_type=metric, params={})
        client.create_index(COLLECTION_NAME, ip, timeout=180.0, sync=True)
        logger.info(f"[Milvus] (embedding) 인덱스 생성 완료")

    # 2) 스파스 벡터 인덱스(text_sparse)
    try:
        idx_sparse = client.list_indexes(collection_name=COLLECTION_NAME, field_name="text_sparse")
    except Exception:
        idx_sparse = []
    if not idx_sparse:
        logger.info(f"[Milvus] (text_sparse) 인덱스 생성 시작")
        ip2 = client.prepare_index_params()
        try:
            # 최신 PyMilvus: metric_type 없이도 동작
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", params={})
        except TypeError:
            # 일부 버전은 metric_type이 필요할 수 있음
            ip2.add_index("text_sparse", "SPARSE_INVERTED_INDEX", metric_type="BM25", params={})
        client.create_index(COLLECTION_NAME, ip2, timeout=180.0, sync=True)
        logger.info(f"[Milvus] (text_sparse) 인덱스 생성 완료")

    # 인덱스 준비 후 로드(이미 로드되어 있으면 내렸다가 다시 올림)
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    client.load_collection(collection_name=COLLECTION_NAME)
    logger.info(f"[Milvus] 컬렉션 로드 완료: {COLLECTION_NAME}")
    
    logger.info(f"[Milvus] 컬렉션 및 인덱스 준비 완료: {COLLECTION_NAME}")


# -------------------------------------------------
# 1) PDF → 텍스트 추출 (작업유형별 보안레벨 동시 산정)
# -------------------------------------------------
async def extract_pdfs():
    from tqdm import tqdm  # type: ignore

    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 규칙 로드
    all_rules = get_security_level_rules_all()  # {task: {"maxLevel":N, "levels":{...}}}

    # 이전 메타 로드
    prev_meta: Dict[str, Dict] = {}
    if META_JSON_PATH.exists():
        try:
            prev_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read META JSON; recreating.")

    # 중복 제거 로직: 파일명 마지막 토큰이 날짜/버전 숫자면 최신만 유지
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

    raw_files = [p for p in RAW_DATA_DIR.rglob("*") if p.is_file() and _ext(p) in SUPPORTED_EXTS]

    # base(문서ID 유사)별로 버전 후보 묶기: (Path, date_num)
    grouped: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for p in raw_files:
        base, date_num = _extract_base_and_date(p)
        grouped[base].append((p, date_num))

    kept, removed = [], []
    for base, lst in grouped.items():
        # lst 원소는 (Path, date_num)
        lst_sorted = sorted(
            lst,
            key=lambda it: (it[1], it[0].stat().st_mtime, len(it[0].name))  # date_num → mtime → 이름 길이
        )
        keep_path = lst_sorted[-1][0]          # 최신 후보의 Path
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
            "meta_path": str(META_JSON_PATH),
            "deduplicated": {"removedCount": len(removed), "removed": removed},
        }

    new_meta: Dict[str, Dict] = {}
    for src in tqdm(kept, desc="문서 전처리"):
        try:
            text, tables = _extract_any(src)
            
            # 작업유형별 보안 레벨 (본문+표 모두 포함해서 판정)
            whole_for_level = text + "\n\n" + "\n\n".join(t.get("text","") for t in (tables or []))
            sec_map = {task: _determine_level_for_task(
                whole_for_level, all_rules.get(task, {"maxLevel": 1, "levels": {}})
            ) for task in TASK_TYPES}

            max_sec = max(sec_map.values()) if sec_map else 1
            sec_folder = f"securityLevel{int(max_sec)}"

            rel_from_raw = src.relative_to(RAW_DATA_DIR)
            # 원본 그대로 보관
            dest_rel = Path(sec_folder) / rel_from_raw
            dest_abs = LOCAL_DATA_ROOT / dest_rel
            dest_abs.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dest_abs)
            except Exception:
                logger.exception("Failed to copy file: %s", dest_abs)

            # 추출 텍스트 저장 (확장자는 .txt로 통일)
            txt_rel = dest_rel.with_suffix(".txt")
            (EXTRACTED_TEXT_DIR / txt_rel).parent.mkdir(parents=True, exist_ok=True)
            (EXTRACTED_TEXT_DIR / txt_rel).write_text(text, encoding="utf-8")

            # doc_id/version 유추
            stem = rel_from_raw.stem
            doc_id, version = _parse_doc_version(stem)

            info = {
                "chars": len(text),
                "lines": len(text.splitlines()),
                "preview": (_clean_text(text[:200].replace("\n"," ")) + "…") if text else "",
                "security_levels": sec_map,  # 작업유형별 보안레벨
                "doc_id": doc_id,
                "version": version,
                "tables": tables or [],  # ★ 표 정보 추가
                "sourceExt": _ext(src),  # 원본 확장자 기록
            }
            new_meta[str(dest_rel)] = info

        except Exception as e:
            logger.exception("Failed to process: %s", src)
            try:
                rel_from_raw = src.relative_to(RAW_DATA_DIR)
                dest_rel = Path("securityLevel1") / rel_from_raw
                new_meta[str(dest_rel)] = {"error": str(e)}
            except Exception:
                new_meta[src.name] = {"error": str(e)}

    META_JSON_PATH.write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "message": "문서 추출 완료",
        "file_count": len(kept),
        "meta_path": str(META_JSON_PATH),
        "deduplicated": {"removedCount": len(removed), "removed": removed},
    }


def _parse_doc_version(stem: str) -> Tuple[str, int]:
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0


# -------------------------------------------------
# 2) 인제스트 (bulk)
#   - 작업유형별로 동일 청크를 각각 저장(task_type, security_level 분리)
# -------------------------------------------------
async def ingest_embeddings(model_key: str | None = None, chunk_size: int | None = None, overlap: int | None = None, target_tasks: list[str] | None = None, collection_name: str = COLLECTION_NAME):
    # rag_settings 단일 소스
    s = get_vector_settings()
    MAX_TOKENS = int(s["chunkSize"])
    OVERLAP = int(s["overlap"])

    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}

    # 모델/검색 설정 로드(모델키 우선순위: 인자 > settings)
    settings = get_vector_settings()
    eff_model_key = model_key or settings["embeddingModel"]

    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])

    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    # 모든 TASK_TYPES 대상으로 고정
    tasks = list(TASK_TYPES)

    total_inserted = 0
    for txt_path in EXTRACTED_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(EXTRACTED_TEXT_DIR)
        # 다양한 확장자를 시도하여 메타 키 찾기
        cands = [rel_txt.with_suffix(ext).as_posix() for ext in SUPPORTED_EXTS]
        meta_key = next((k for k in cands if k in meta), None)
        if not meta_key:
            continue
        entry = meta[meta_key]
        tables = entry.get("tables", [])
        sec_map = entry.get("security_levels", {}) or {}
        doc_id = entry.get("doc_id")
        version = entry.get("version", 0)
        if not doc_id or version == 0:
            _id, _ver = _parse_doc_version(rel_txt.stem)
            doc_id = doc_id or _id
            version = version or _ver
            entry["doc_id"] = doc_id
            entry["version"] = version

        # 이전 동일 doc_id/version 데이터 삭제(작업유형 전체)
        try:
            client.delete(
                COLLECTION_NAME,
                filter=f"doc_id == '{doc_id}' && version <= {int(version)}",
            )
        except Exception:
            pass

        # 텍스트 로드/청크화
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        # 작업유형별 삽입
        batch: List[Dict] = []
        tables = entry.get("tables", [])  # [{page,bbox,text}, ...]
        TABLE_MARK = "[[TABLE"
        
        for task in tasks:
            lvl = int(sec_map.get(task, 1))
            
            # 본문 청크 삽입 (기존 그대로)
            for idx, c in enumerate(chunks):
                vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_txt),
                    "chunk_idx": int(idx),
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(version),
                    "text": c,
                })
                if len(batch) >= 128:
                    client.insert(COLLECTION_NAME, batch)
                    total_inserted += len(batch)
                    batch = []
            
            # ★ 표 청크 삽입 (절대 분할하지 않음)
            base_idx = len(chunks)  # 표는 본문 뒤 인덱스부터
            for t_i, t in enumerate(tables):
                md = (t.get("text") or "").strip()
                if not md:
                    continue
                page = int(t.get("page", 0))
                bbox = t.get("bbox") or []
                bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                table_text = f"{TABLE_MARK} page={page} bbox={bbox_str}]]\n{md}"
                vec = _embed_text(tok, model, device, table_text, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_txt),             # 경로는 동일 txt 기준 유지
                    "chunk_idx": int(base_idx + t_i), # 본문 뒤로 이어붙임
                    "task_type": task,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(version),
                    "text": table_text,               # ★ 표 마커+마크다운
                })
                if len(batch) >= 128:
                    client.insert(COLLECTION_NAME, batch)
                    total_inserted += len(batch)
                    batch = []
        if batch:
            client.insert(COLLECTION_NAME, batch)
            total_inserted += len(batch)

    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")
    META_JSON_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"message": "Ingest 완료(Milvus Server)", "inserted_chunks": total_inserted}


# -------------------------------------------------
# 2-1) 단일 파일 인제스트(선택 작업유형)
# -------------------------------------------------
async def ingest_single_pdf(req: SinglePDFIngestRequest):
    try:
        from repository.documents import insert_workspace_document
    except Exception:
        insert_workspace_document = None

    file_path = Path(req.pdf_path)
    if not file_path.exists():
        return {"error": f"파일 경로를 찾을 수 없습니다: {file_path}"}

    # 지원되지 않는 확장자 체크
    if _ext(file_path) not in SUPPORTED_EXTS:
        return {"error": f"지원되지 않는 파일 형식입니다: {_ext(file_path)}"}

    # 텍스트 생성 및 메타 갱신
    if META_JSON_PATH.exists():
        meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
    else:
        meta = {}

    # 다중 확장자 지원으로 텍스트 추출
    text_all, table_blocks_all = _extract_any(file_path)

    all_rules = get_security_level_rules_all()
    # 본문+표 모두 포함해서 보안레벨 판정
    whole_for_level = text_all + "\n\n" + "\n\n".join(t.get("text","") for t in (table_blocks_all or []))
    sec_map = {
        task: _determine_level_for_task(
            whole_for_level, all_rules.get(task, {"maxLevel": 1, "levels": {}})
        )
        for task in TASK_TYPES
    }
    max_sec = max(sec_map.values()) if sec_map else 1
    sec_folder = f"securityLevel{int(max_sec)}"

    rel_file = Path(sec_folder) / file_path.name
    # 저장 경로: local_data(원본 파일 복사) + extracted_texts(텍스트)
    (LOCAL_DATA_ROOT / rel_file).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, LOCAL_DATA_ROOT / rel_file)
    txt_path = EXTRACTED_TEXT_DIR / rel_file.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text_all, encoding="utf-8")

    doc_id, ver = _parse_doc_version(file_path.stem)
    meta[str(rel_file)] = {
        "chars": len(text_all),
        "lines": len(text_all.splitlines()),
        "preview": (_clean_text(text_all[:200].replace("\n", " ")) + "…") if text_all else "",
        "security_levels": sec_map,
        "doc_id": doc_id,
        "version": ver,
        "tables": table_blocks_all or [],  # ★ 표 정보 추가
        "sourceExt": _ext(file_path),  # 원본 확장자 기록
    }
    META_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_JSON_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 인제스트(선택 작업유형)
    settings = get_vector_settings()
    tok, model, device = _load_embedder(settings["embeddingModel"])
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])
    client = _client()
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    s = get_vector_settings()
    MAX_TOKENS, OVERLAP = int(s["chunkSize"]), int(s["overlap"])
    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    # 기존 삭제
    try:
        client.delete(
            COLLECTION_NAME, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}"
        )
    except Exception:
        pass

    tasks = req.task_types or list(TASK_TYPES)
    chunks = chunk_text(text_all)
    batch, cnt = [], 0
    TABLE_MARK = "[[TABLE"
    
    for task in tasks:
        lvl = int(sec_map.get(task, 1))
        
        # 본문 청크 삽입
        for idx, c in enumerate(chunks):
            vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
            batch.append({
                "embedding": vec.tolist(),
                "path": str(rel_file.with_suffix(".txt")),
                "chunk_idx": int(idx),
                "task_type": task,
                "security_level": lvl,
                "doc_id": str(doc_id),
                "version": int(ver),
                "text": c,
            })
            if len(batch) >= 128:
                client.insert(COLLECTION_NAME, batch)
                cnt += len(batch)
                batch = []
        
        # ★ 표 청크 삽입 (절대 분할하지 않음)
        base_idx = len(chunks)  # 표는 본문 뒤 인덱스부터
        for t_i, t in enumerate(table_blocks_all):
            md = (t.get("text") or "").strip()
            if not md:
                continue
            page = int(t.get("page", 0))
            bbox = t.get("bbox") or []
            bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
            table_text = f"{TABLE_MARK} page={page} bbox={bbox_str}]]\n{md}"
            vec = _embed_text(tok, model, device, table_text, max_len=MAX_TOKENS)
            batch.append({
                "embedding": vec.tolist(),
                "path": str(rel_file.with_suffix(".txt")),
                "chunk_idx": int(base_idx + t_i),
                "task_type": task,
                "security_level": lvl,
                "doc_id": str(doc_id),
                "version": int(ver),
                "text": table_text,  # ★ 표 마커+마크다운
            })
            if len(batch) >= 128:
                client.insert(COLLECTION_NAME, batch)
                cnt += len(batch)
                batch = []
    if batch:
        client.insert(COLLECTION_NAME, batch)
        cnt += len(batch)

    if req.workspace_id is not None and insert_workspace_document:
        try:
            insert_workspace_document(
                doc_id=str(doc_id),
                filename=rel_file.name,
                docpath=str(rel_file),
                workspace_id=int(req.workspace_id),
                metadata={
                    "securityLevels": sec_map,
                    "chunks": int(cnt),
                    "isUserUpload": True,
                },
            )
        except Exception:
            pass

    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        pass
    _ensure_collection_and_index(client, emb_dim, metric="IP")

    return {
        "message": f"단일 파일 인제스트 완료(Milvus Server) - {_ext(file_path)}",
        "doc_id": doc_id,
        "version": ver,
        "chunks": cnt,
        "sourceExt": _ext(file_path),
    }


# -------------------------------------------------
# 3) 검색 (vector / hybrid)
#   - task_type 필터 + security_level 제한
#   - hybrid: 벡터 topK*α 후보에 대해 간이 BM25 후처리 리랭크
# -------------------------------------------------
def _bm25_like_score(query: str, doc: str, k1: float = 1.2, b: float = 0.75) -> float:
    # 후보군 소규모 리랭크용 간단 BM25 대용(문서 집합이 작을 때만)
    # 토크나이징 매우 단순화(공백 기준)
    q_terms = [w for w in query.lower().split() if w]
    d_terms = [w for w in doc.lower().split() if w]
    if not q_terms or not d_terms:
        return 0.0
    d_len = len(d_terms)
    tf = Counter(d_terms)
    # IDF는 후보 집합 크기를 사용하기 어려워 고정치에 완화 가중
    score = 0.0
    avgdl = max(1.0, d_len)  # 후보 단일 문서 기준
    for t in set(q_terms):
        f = tf.get(t, 0)
        if f == 0:
            continue
        # 완화 IDF(상수): log(1 + 1/freq) 대신 상수 1.5 사용(경험적)
        idf = 1.5
        denom = f + k1 * (1 - b + b * (d_len / avgdl))
        score += idf * ((f * (k1 + 1)) / (denom if denom != 0 else 1))
    return float(score)


async def search_documents(req: RAGSearchRequest, search_type_override: Optional[str] = None,
                           collection_name: str = COLLECTION_NAME) -> Dict:
    t0 = time.perf_counter()
    if req.task_type not in TASK_TYPES:
        return {
            "error": f"invalid task_type: {req.task_type}. choose one of {TASK_TYPES}"
        }

    settings = get_vector_settings()
    model_key = req.model or settings["embeddingModel"]
    raw_st = (search_type_override or settings.get("searchType") or "").lower()
    # alias normalization: 'semantic'/'sementic' -> 'vector'; default 'hybrid' if empty
    search_type = (raw_st.replace("semantic", "vector").replace("sementic", "vector") or "hybrid")

    tok, model, device = await _get_or_load_embedder_async(model_key)
    q_emb = _embed_text(tok, model, device, req.query)
    client = _client()
    _ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP")

    if COLLECTION_NAME not in client.list_collections():
        return {"error": "컬렉션이 없습니다. 먼저 데이터 저장(인제스트)을을 수행하세요."}

    # 공통 파라미터
    base_limit = int(req.top_k)
    candidate = min(50, max(base_limit, base_limit * 4))
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    def _dense_search(limit=candidate):
        return client.search(
            collection_name=COLLECTION_NAME,
            data=[q_emb.tolist()],
            anns_field="embedding",
            limit=int(limit),
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
            filter=filter_expr,
        )

    def _sparse_search(query_text: str, limit=candidate):
        """
        BM25 Function(text->text_sparse)이 붙어 있으면 서버가 스파스 점수를 계산한다.
        최신 pymilvus에서는 anns_field='text_sparse', data=['쿼리 문자열'] 형태를 지원.
        일부 버전에서 미지원이면 예외 발생 → 폴백으로 빈 결과 반환.
        """
        try:
            return client.search(
                collection_name=COLLECTION_NAME,
                data=[query_text],
                anns_field="text_sparse",
                limit=int(limit),
                search_params={"params": {}},
                output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning(f"[Milvus] sparse search unavailable: {e}")
            return [[]]

    def _load_snippet(
        path: str, cidx: int, max_tokens: int = 512, overlap: int = 64
    ) -> str:
        try:
            full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        except Exception:
            return ""
        words = full_txt.split()
        if not words:
            return ""
        start = cidx * (max_tokens - overlap)
        # 보존: 추출 시와 동일 슬라이딩 윈도우는 아니지만 근사 스니펫 제공
        snippet = " ".join(words[start : start + max_tokens]).strip()
        return snippet or " ".join(words[:max_tokens]).strip()

    # === 분기: 검색 방식 ===
    hits_raw = []
    TABLE_MARK = "[[TABLE"
    
    if search_type == "vector":
        results = _dense_search(limit=candidate)
        for hit in results[0]:
            if isinstance(hit, dict):
                ent = hit.get("entity", {})
                ent_text = ent.get("text")  # ★ 추가
                path = ent.get("path")
                cidx = int(ent.get("chunk_idx", 0))
                ttype = ent.get("task_type")
                lvl = int(ent.get("security_level", 1))
                doc_id = ent.get("doc_id")
                score_vec = float(hit.get("distance", 0.0))
            else:
                ent = hit.entity
                ent_text = getattr(ent, "get", lambda _k: None)("text") if hasattr(ent, "get") else None
                path = hit.entity.get("path")
                cidx = int(hit.entity.get("chunk_idx", 0))
                ttype = hit.entity.get("task_type")
                lvl = int(hit.entity.get("security_level", 1))
                doc_id = hit.entity.get("doc_id")
                score_vec = float(hit.score)
            
            # 스니펫 결정 로직: 표면 저장된 텍스트 그대로, 아니면 기존 로직
            if isinstance(ent_text, str) and ent_text.startswith(TABLE_MARK):
                snippet = ent_text  # ★ 표는 저장된 마크다운 그대로
            else:
                snippet = _load_snippet(path, cidx)
                
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype,
                "security_level": lvl, "doc_id": doc_id,
                "score_vec": score_vec, "score_sparse": 0.0, "snippet": snippet
            })
    else:
        # hybrid / bm25: 덴스 + 스파스 각각 검색
        res_dense = _dense_search(limit=candidate)
        res_sparse = _sparse_search(req.query, limit=candidate)

        def _collect(res, is_dense: bool):
            out = []
            for hit in (res[0] if res and len(res) > 0 else []):
                if isinstance(hit, dict):
                    ent = hit.get("entity", {})
                    ent_text = ent.get("text")  # ★ 추가
                    path = ent.get("path")
                    cidx = int(ent.get("chunk_idx", 0))
                    ttype = ent.get("task_type")
                    lvl = int(ent.get("security_level", 1))
                    doc_id = ent.get("doc_id")
                    score = float(hit.get("distance", 0.0))
                else:
                    ent = hit.entity
                    ent_text = getattr(ent, "get", lambda _k: None)("text") if hasattr(ent, "get") else None
                    path = hit.entity.get("path")
                    cidx = int(hit.entity.get("chunk_idx", 0))
                    ttype = hit.entity.get("task_type")
                    lvl = int(hit.entity.get("security_level", 1))
                    doc_id = hit.entity.get("doc_id")
                    score = float(hit.score)
                out.append(((path, cidx, ttype, lvl, doc_id, ent_text), score))  # ★ ent_text 추가
            return out

        dense_list = _collect(res_dense, True)
        sparse_list = _collect(res_sparse, False)

        # RRF 결합 폴백
        rrf: dict[tuple, float] = {}
        text_map: dict[tuple, str] = {}  # ★ 텍스트 매핑 추가
        K = 60.0
        for rank, (key, _s) in enumerate(dense_list, start=1):
            key_short = key[:5]  # (path, cidx, ttype, lvl, doc_id)
            rrf[key_short] = rrf.get(key_short, 0.0) + 1.0 / (K + rank)
            if len(key) > 5:  # ent_text가 있으면 저장
                text_map[key_short] = key[5]
        for rank, (key, _s) in enumerate(sparse_list, start=1):
            key_short = key[:5]  # (path, cidx, ttype, lvl, doc_id)
            rrf[key_short] = rrf.get(key_short, 0.0) + 1.0 / (K + rank)
            if len(key) > 5:  # ent_text가 있으면 저장
                text_map[key_short] = key[5]

        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate]
        for (path, cidx, ttype, lvl, doc_id), fused in merged:
            # 스니펫 결정 로직
            ent_text = text_map.get((path, cidx, ttype, lvl, doc_id))
            if isinstance(ent_text, str) and ent_text.startswith(TABLE_MARK):
                snippet = ent_text  # ★ 표는 저장된 마크다운 그대로
            else:
                snippet = _load_snippet(path, cidx)
            
            s_vec = next((s for (k, s) in dense_list if k[:5] == (path, cidx, ttype, lvl, doc_id)), 0.0)
            s_spa = next((s for (k, s) in sparse_list if k[:5] == (path, cidx, ttype, lvl, doc_id)), 0.0)
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype,
                "security_level": lvl, "doc_id": doc_id,
                "score_vec": float(s_vec), "score_sparse": float(s_spa),
                "score_fused": float(fused),
                "snippet": snippet
            })

    # 최종 스코어/정렬
    if search_type == "bm25":
        for h in hits_raw:
            h["score"] = h.get("score_sparse", 0.0) or h.get("score_vec", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]
    elif search_type == "hybrid":
        if any("score_fused" in h for h in hits_raw):
            for h in hits_raw:
                h["score"] = h.get("score_fused", 0.0)
        else:
            for h in hits_raw:
                h["score"] = 0.5 * h.get("score_vec", 0.0) + 0.5 * h.get("score_sparse", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]
    else:  # vector
        for h in hits_raw:
            h["score"] = h.get("score_vec", 0.0)
        hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[:base_limit]

    # 프롬프트 컨텍스트 생성
    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."

    elapsed = round(time.perf_counter() - t0, 4)

    # query_logs 삭제: INSERT 제거
    return {
        "elapsed_sec": elapsed,
        "settings_used": {"model": model_key, "searchType": search_type},
        "hits": [
            {
                "score": float(h["score"]),
                "path": h["path"],
                "chunk_idx": int(h["chunk_idx"]),
                "task_type": h["task_type"],
                "security_level": int(h["security_level"]),
                "doc_id": h.get("doc_id"),
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
    }


async def execute_search(
    question: str,
    top_k: int = 5,
    security_level: int = 1,
    source_filter: Optional[List[str]] = None,
    task_type: str = "qna",
    model_key: Optional[str] = None,
    search_type: Optional[str] = None,
) -> Dict:
    req = RAGSearchRequest(
        query=question,
        top_k=top_k,
        user_level=security_level,
        task_type=task_type,
        model=model_key,
    )
    res = await search_documents(req, search_type_override=search_type)
    # Build check_file BEFORE optional source_filter so it reflects original candidates
    check_files: List[str] = []
    try:
        for h in res.get("hits", []):
            # Prefer doc_id when available; fallback to path-derived filename
            doc_id_val = h.get("doc_id")
            if doc_id_val:
                check_files.append(f"{str(doc_id_val)}.pdf")
                continue
            p = Path(h.get("path", ""))
            if str(p):
                check_files.append(p.with_suffix(".pdf").name)
    except Exception:
        pass

    if source_filter and "hits" in res:
        names = {Path(n).stem for n in source_filter}
        res["hits"] = [h for h in res["hits"] if Path(h["path"]).stem in names]

    res["check_file"] = sorted(list(set(check_files)))
    return res


# -------------------------------------------------
# 4) 관리 유틸
# -------------------------------------------------
async def delete_db():
    # 모델 캐시 클리어
    _invalidate_embedder_cache()

    client = _client()
    cols = client.list_collections()
    for c in cols:
        client.drop_collection(c)
    return {"message": "삭제 완료(Milvus Server)", "dropped_collections": cols}


async def list_indexed_files(
    limit: int = 16384,
    offset: int = 0,
    query: Optional[str] = None,
    task_type: Optional[str] = None,
):
    limit = max(1, min(limit, 16384))
    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        return []

    flt = ""
    if task_type and task_type in TASK_TYPES:
        flt = f"task_type == '{task_type}'"
    try:
        rows = client.query(
            collection_name=COLLECTION_NAME,
            filter=flt,
            output_fields=["path", "chunk_idx", "security_level", "task_type"],
            limit=limit,
            offset=offset,
            consistency_level="Strong",
        )
    except Exception:
        rows = []

    counts: Dict[Tuple[str, str], int] = defaultdict(int)  # (path, task_type) -> chunks
    level_map: Dict[Tuple[str, str], int] = {}
    for r in rows:
        path = r.get("path") if isinstance(r, dict) else r["path"]
        ttype = r.get("task_type") if isinstance(r, dict) else r["task_type"]
        lvl = int(
            (r.get("security_level") if isinstance(r, dict) else r["security_level"])
            or 1
        )
        key = (path, ttype)
        counts[key] += 1
        level_map.setdefault(key, lvl)

    items = []
    for (path, ttype), cnt in counts.items():
        txt_rel = Path(path)
        pdf_rel = txt_rel.with_suffix(".pdf")
        file_name = pdf_rel.name
        txt_abs = EXTRACTED_TEXT_DIR / txt_rel
        try:
            stat = txt_abs.stat()
            size = stat.st_size
            indexed_at = now_kst_string()
        except FileNotFoundError:
            size = None
            indexed_at = None
        items.append(
            {
                "taskType": ttype,
                "fileName": file_name,
                "filePath": str(pdf_rel),
                "chunkCount": int(cnt),
                "indexedAt": indexed_at,
                "fileSize": size,
                "securityLevel": int(level_map.get((path, ttype), 1)),
            }
        )

    if query:
        q = str(query)
        items = [it for it in items if q in it["fileName"]]
    return items


async def delete_files_by_names(file_names: List[str], task_type: Optional[str] = None, collection_name: str = COLLECTION_NAME):
    """
    파일명(= doc_id stem) 배열을 받아 벡터 DB에서 삭제.
    - task_type 가 None 이면 모든 작업유형(doc_gen/summary/qna)에서 삭제 (기존 동작과 동일)
    - task_type 가 지정되면 해당 작업유형 레코드만 삭제
    """
    if not file_names:
        return {"deleted": 0, "requested": 0}

    try:
        from repository.documents import delete_workspace_documents_by_filenames
    except Exception:
        delete_workspace_documents_by_filenames = None

    client = _client()
    if COLLECTION_NAME not in client.list_collections():
        deleted_sql = None
        if delete_workspace_documents_by_filenames:
            deleted_sql = delete_workspace_documents_by_filenames(file_names)
        return {"deleted": 0, "deleted_sql": deleted_sql, "requested": len(file_names)}

    # 로드 보장
    try:
        client.load_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass

    # 유효한 task_type 인지 검증
    task_filter = ""
    if task_type:
        if task_type not in TASK_TYPES:
            return {
                "deleted": 0,
                "requested": len(file_names),
                "error": f"invalid taskType: {task_type}",
            }
        task_filter = f" && task_type == '{task_type}'"

    deleted_total = 0
    per_file: dict[str, int] = {}

    for name in file_names:
        stem = Path(name).stem
        # Align fileName -> doc_id by stripping version suffix if present
        try:
            base_id, _ver = _parse_doc_version(stem)
        except Exception:
            base_id = stem
        try:
            # doc_id == 'stem' [&& task_type == 'xxx']
            filt = f"doc_id == '{base_id}'{task_filter}"
            client.delete(collection_name=COLLECTION_NAME, filter=filt)
            deleted_total += 1
            per_file[name] = per_file.get(name, 0) + 1
        except Exception:
            logger.exception("Failed to delete from Milvus for file: %s", name)
            per_file[name] = per_file.get(name, 0)

    # Ensure deletion is visible to subsequent queries (file lists/overview)
    try:
        client.flush(COLLECTION_NAME)
    except Exception:
        logger.exception("Failed to flush Milvus after deletion")
    # Force reload to avoid any stale cache/state on the server side
    try:
        client.release_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    try:
        client.load_collection(collection_name=COLLECTION_NAME)
    except Exception:
        logger.exception("Failed to reload collection after deletion")

    deleted_sql = None
    if delete_workspace_documents_by_filenames:
        try:
            # SQL은 작업유형 구분이 없다고 가정(기존 그대로)
            deleted_sql = delete_workspace_documents_by_filenames(file_names)
        except Exception:
            logger.exception("Failed to delete workspace documents in SQL")
            deleted_sql = None

    return {
        "deleted": deleted_total,  # 요청 파일 기준 성공 건수(작업유형 기준 단순 카운트)
        "deleted_sql": deleted_sql,
        "requested": len(file_names),
        "taskType": task_type,
        "perFile": per_file,  # 파일별 처리현황
    }


async def list_indexed_files_overview(collection_name: str = COLLECTION_NAME):
    items = await list_indexed_files(limit=16384, offset=0, query=None, task_type=None)
    # agg: task_type -> level -> count
    agg: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for it in items:
        agg[it["taskType"]][int(it["securityLevel"])] += it["chunkCount"]
    # 보기 좋게 변환
    overview = {
        t: {str(lv): agg[t][lv] for lv in sorted(agg[t].keys())} for t in agg.keys()
    }
    return {"overview": overview, "items": items}


# --- add: 세션 인덱스 로드/저장 + 컬렉션명 ---
def _load_sessions_index() -> dict:
    VAL_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    if SESSIONS_INDEX_PATH.exists():
        try:
            return json.loads(SESSIONS_INDEX_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to read sessions index")
    return {}

def _save_sessions_index(idx: dict):
    VAL_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    SESSIONS_INDEX_PATH.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")

def _session_collection_name(sid: str) -> str:
    return f"{COLLECTION_NAME}__sess__{sid}"

# --- add: 세션 생성/조회/삭제 ---
def create_test_session() -> Dict:
    idx = _load_sessions_index()
    sid = uuid.uuid4().hex[:12]
    sess_dir = (VAL_SESSION_ROOT / sid).resolve()
    sess_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "sid": sid,
        "dir": str(sess_dir),
        "collection": _session_collection_name(sid),
        "createdAt": now_kst_string(),
    }
    idx[sid] = meta
    _save_sessions_index(idx)
    return meta

def get_test_session(sid: str) -> Optional[Dict]:
    idx = _load_sessions_index()
    return idx.get(sid)

async def drop_test_session(sid: str) -> Dict:
    idx = _load_sessions_index()
    meta = idx.pop(sid, None)
    if not meta:
        return {"success": False, "error": "invalid sid"}

    # 1) Milvus 컬렉션 드롭
    try:
        client = _client()
        coll = meta.get("collection") or _session_collection_name(sid)
        if coll in client.list_collections():
            client.drop_collection(coll)
    except Exception:
        logger.exception("[test-session] drop collection failed")

    # 2) 로컬 디렉토리 삭제(업로드 PDF)
    try:
        p = Path(meta["dir"])
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        logger.exception("[test-session] remove session dir failed")

    # 3) 세션 텍스트 폴더(EXTRACTED_TEXT_DIR/__sessions__/sid) 삭제
    try:
        sess_txt = EXTRACTED_TEXT_DIR / "__sessions__" / sid
        if sess_txt.exists():
            shutil.rmtree(sess_txt, ignore_errors=True)
    except Exception:
        logger.exception("[test-session] remove session texts failed")

    _save_sessions_index(idx)
    return {"success": True, "sid": sid, "dropped": True}

# --- add: ingest_test_pdfs ---
async def ingest_test_pdfs(sid: str, pdf_paths: List[str], task_types: Optional[List[str]] = None):
    """
    업로드된 파일들을 세션 전용 컬렉션에 인제스트.
    다중 확장자 지원 (pdf, txt, docx, pptx, csv, xlsx, xls, doc, ppt)
    텍스트는 EXTRACTED_TEXT_DIR/__sessions__/{sid}/securityLevelN/ 아래에 저장하여
    기존 _load_snippet 경로 로직을 그대로 활용.
    """
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}

    # 설정/모델 로드
    settings = get_vector_settings()
    eff_model_key = settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(_embed_text(tok, model, device, "probe").shape[0])

    client = _client()
    coll = meta.get("collection") or _session_collection_name(sid)
    _ensure_collection_and_index_for(client, collection_name=coll, emb_dim=emb_dim, metric="IP")

    s = get_vector_settings()
    MAX_TOKENS, OVERLAP = int(s["chunkSize"]), int(s["overlap"])

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    # 보안 레벨 규칙
    all_rules = get_security_level_rules_all()

    # 세션 텍스트 루트
    sess_txt_root = EXTRACTED_TEXT_DIR / "__sessions__" / sid
    sess_txt_root.mkdir(parents=True, exist_ok=True)

    tasks = task_types or list(TASK_TYPES)
    total = 0

    for p in pdf_paths:
        p = str(p)
        file_path = Path(p)
        
        # 지원되지 않는 확장자는 건너뛰기
        if _ext(file_path) not in SUPPORTED_EXTS:
            logger.warning(f"[test-ingest] Unsupported file type: {file_path}")
            continue
            
        try:
            # 다중 확장자 지원으로 텍스트 추출
            file_text, table_blocks_all = _extract_any(file_path)
        except Exception as e:
            logger.exception("[test-ingest] read failed: %s", p)
            continue

        # 작업유형별 보안 레벨 산정 (본문+표 모두 포함)
        whole_for_level = file_text + "\n\n" + "\n\n".join(t.get("text","") for t in (table_blocks_all or []))
        sec_map = {t: _determine_level_for_task(whole_for_level, all_rules.get(t, {"maxLevel": 1, "levels": {}})) for t in TASK_TYPES}
        max_sec = max(sec_map.values()) if sec_map else 1
        sec_folder = f"securityLevel{int(max_sec)}"

        # 세션 텍스트 파일 저장 (EXTRACTED_TEXT_DIR 기준 상대 경로 구성)
        # 확장자를 .txt로 통일
        rel_txt = Path("__sessions__") / sid / sec_folder / Path(p).with_suffix(".txt").name
        abs_txt = EXTRACTED_TEXT_DIR / rel_txt
        abs_txt.parent.mkdir(parents=True, exist_ok=True)
        abs_txt.write_text(file_text, encoding="utf-8")

        # doc_id/version 유추
        stem = Path(p).stem
        doc_id, ver = _parse_doc_version(stem)

        # 기존 동일 문서 제거
        try:
            client.delete(coll, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}")
        except Exception:
            pass

        # 청크 → 삽입
        chunks = chunk_text(file_text)
        batch: List[Dict] = []
        TABLE_MARK = "[[TABLE"
        
        for t in tasks:
            lvl = int(sec_map.get(t, 1))
            
            # 본문 청크 삽입
            for idx, c in enumerate(chunks):
                vec = _embed_text(tok, model, device, c, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_txt.as_posix()),
                    "chunk_idx": int(idx),
                    "task_type": t,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "text": c,
                })
                if len(batch) >= 128:
                    client.insert(coll, batch)
                    total += len(batch)
                    batch = []
            
            # ★ 표 청크 삽입 (절대 분할하지 않음)
            base_idx = len(chunks)  # 표는 본문 뒤 인덱스부터
            for t_i, table in enumerate(table_blocks_all):
                md = (table.get("text") or "").strip()
                if not md:
                    continue
                page = int(table.get("page", 0))
                bbox = table.get("bbox") or []
                bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                table_text = f"{TABLE_MARK} page={page} bbox={bbox_str}]]\n{md}"
                vec = _embed_text(tok, model, device, table_text, max_len=MAX_TOKENS)
                batch.append({
                    "embedding": vec.tolist(),
                    "path": str(rel_txt.as_posix()),
                    "chunk_idx": int(base_idx + t_i),
                    "task_type": t,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "text": table_text,  # ★ 표 마커+마크다운
                })
                if len(batch) >= 128:
                    client.insert(coll, batch)
                    total += len(batch)
                    batch = []
        if batch:
            client.insert(coll, batch)
            total += len(batch)

    try:
        client.flush(coll)
    except Exception:
        pass
    _ensure_collection_and_index_for(client, collection_name=coll, emb_dim=emb_dim, metric="IP")

    return {"message": "세션 인제스트 완료", "sid": sid, "inserted_chunks": total}

# --- add: search_documents_test ---
async def search_documents_test(req: RAGSearchRequest, sid: str, search_type_override: Optional[str] = None) -> Dict:
    """
    세션 전용 컬렉션에서만 검색 (기존 search_documents의 세션 버전)
    """
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}

    t0 = time.perf_counter()
    if req.task_type not in TASK_TYPES:
        return {"error": f"invalid task_type: {req.task_type}. choose one of {TASK_TYPES}"}

    settings = get_vector_settings()
    model_key = req.model or settings["embeddingModel"]
    raw_st = (search_type_override or settings.get("searchType") or "").lower()
    search_type = (raw_st.replace("semantic", "vector").replace("sementic", "vector") or "hybrid")

    tok, model, device = await _get_or_load_embedder_async(model_key)
    q_emb = _embed_text(tok, model, device, req.query)

    client = _client()
    coll = meta.get("collection") or _session_collection_name(sid)
    _ensure_collection_and_index_for(client, collection_name=coll, emb_dim=len(q_emb), metric="IP")
    if coll not in client.list_collections():
        return {"error": "세션 컬렉션이 없습니다. 먼저 인제스트 하세요."}

    base_limit = int(req.top_k)
    candidate = min(50, max(base_limit, base_limit * 4))
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    def _dense_search(limit=candidate):
        return client.search(
            collection_name=coll,
            data=[q_emb.tolist()],
            anns_field="embedding",
            limit=int(limit),
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
            filter=filter_expr,
        )

    def _sparse_search(query_text: str, limit=candidate):
        try:
            return client.search(
                collection_name=coll,
                data=[query_text],
                anns_field="text_sparse",
                limit=int(limit),
                search_params={"params": {}},
                output_fields=["path", "chunk_idx", "task_type", "security_level", "doc_id", "text"],
                filter=filter_expr,
            )
        except Exception as e:
            logger.warning(f"[Milvus] sparse search unavailable(test): {e}")
            return [[]]

    def _load_snippet_for_session(path: str, cidx: int, max_tokens: int = settings["chunkSize"], overlap: int = settings["overlap"]) -> str:
        # path는 EXTRACTED_TEXT_DIR 기준 상대경로("__sessions__/sid/...")로 저장되어 있음
        try:
            full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        except Exception:
            return ""
        words = full_txt.split()
        if not words:
            return ""
        start = cidx * (max_tokens - overlap)
        snippet = " ".join(words[start : start + max_tokens]).strip()
        return snippet or " ".join(words[:max_tokens]).strip()

    hits_raw = []
    TABLE_MARK = "[[TABLE"
    
    if search_type == "vector":
        res = _dense_search(limit=candidate)
        for hit in res[0]:
            if isinstance(hit, dict):
                ent = hit.get("entity", {})
                ent_text = ent.get("text")  # ★ 추가
                path = ent.get("path")
                cidx = int(ent.get("chunk_idx", 0))
                ttype = ent.get("task_type")
                lvl = int(ent.get("security_level", 1))
                doc_id = ent.get("doc_id")
                score_vec = float(hit.get("distance", 0.0))
            else:
                ent = hit.entity
                ent_text = getattr(ent, "get", lambda _k: None)("text") if hasattr(ent, "get") else None
                path = hit.entity.get("path")
                cidx = int(hit.entity.get("chunk_idx", 0))
                ttype = hit.entity.get("task_type")
                lvl = int(hit.entity.get("security_level", 1))
                doc_id = hit.entity.get("doc_id")
                score_vec = float(hit.score)
            
            # 스니펫 결정 로직: 표면 저장된 텍스트 그대로, 아니면 기존 로직
            if isinstance(ent_text, str) and ent_text.startswith(TABLE_MARK):
                snippet = ent_text  # ★ 표는 저장된 마크다운 그대로
            else:
                snippet = _load_snippet_for_session(path, cidx)
                
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype, "security_level": lvl,
                "doc_id": doc_id, "score_vec": score_vec, "score_sparse": 0.0, "snippet": snippet
            })
    else:
        res_dense = _dense_search(limit=candidate)
        res_sparse = _sparse_search(req.query, limit=candidate)

        def _collect(res):
            out = []
            for hit in (res[0] if res and len(res) > 0 else []):
                if isinstance(hit, dict):
                    ent = hit.get("entity", {})
                    ent_text = ent.get("text")  # ★ 추가
                    path = ent.get("path")
                    cidx = int(ent.get("chunk_idx", 0))
                    ttype = ent.get("task_type")
                    lvl = int(ent.get("security_level", 1))
                    doc_id = ent.get("doc_id")
                    score = float(hit.get("distance", 0.0))
                else:
                    ent = hit.entity
                    ent_text = getattr(ent, "get", lambda _k: None)("text") if hasattr(ent, "get") else None
                    path = hit.entity.get("path")
                    cidx = int(hit.entity.get("chunk_idx", 0))
                    ttype = hit.entity.get("task_type")
                    lvl = int(hit.entity.get("security_level", 1))
                    doc_id = hit.entity.get("doc_id")
                    score = float(hit.score)
                out.append(((path, cidx, ttype, lvl, doc_id, ent_text), score))  # ★ ent_text 추가
            return out

        dense_list = _collect(res_dense)
        sparse_list = _collect(res_sparse)
        rrf: Dict[tuple, float] = {}
        text_map: Dict[tuple, str] = {}  # ★ 텍스트 매핑 추가
        K = 60.0
        for rank, (key, _s) in enumerate(dense_list, start=1):
            key_short = key[:5]  # (path, cidx, ttype, lvl, doc_id)
            rrf[key_short] = rrf.get(key_short, 0.0) + 1.0 / (K + rank)
            if len(key) > 5:  # ent_text가 있으면 저장
                text_map[key_short] = key[5]
        for rank, (key, _s) in enumerate(sparse_list, start=1):
            key_short = key[:5]  # (path, cidx, ttype, lvl, doc_id)
            rrf[key_short] = rrf.get(key_short, 0.0) + 1.0 / (K + rank)
            if len(key) > 5:  # ent_text가 있으면 저장
                text_map[key_short] = key[5]
        merged = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:candidate]
        for (path, cidx, ttype, lvl, doc_id), fused in merged:
            # 스니펫 결정 로직
            ent_text = text_map.get((path, cidx, ttype, lvl, doc_id))
            if isinstance(ent_text, str) and ent_text.startswith(TABLE_MARK):
                snippet = ent_text  # ★ 표는 저장된 마크다운 그대로
            else:
                snippet = _load_snippet_for_session(path, cidx)
            
            s_vec = next((s for (k, s) in dense_list if k[:5] == (path, cidx, ttype, lvl, doc_id)), 0.0)
            s_spa = next((s for (k, s) in sparse_list if k[:5] == (path, cidx, ttype, lvl, doc_id)), 0.0)
            hits_raw.append({
                "path": path, "chunk_idx": cidx, "task_type": ttype, "security_level": lvl, "doc_id": doc_id,
                "score_vec": float(s_vec), "score_sparse": float(s_spa), "score_fused": float(fused), "snippet": snippet
            })

    # 정렬
    if search_type == "bm25":
        for h in hits_raw:
            h["score"] = h.get("score_sparse", 0.0) or h.get("score_vec", 0.0)
    elif search_type == "hybrid":
        if any("score_fused" in h for h in hits_raw):
            for h in hits_raw:
                h["score"] = h.get("score_fused", 0.0)
        else:
            for h in hits_raw:
                h["score"] = 0.5 * h.get("score_vec", 0.0) + 0.5 * h.get("score_sparse", 0.0)
    else:
        for h in hits_raw:
            h["score"] = h.get("score_vec", 0.0)

    hits_sorted = sorted(hits_raw, key=lambda x: x["score"], reverse=True)[: int(req.top_k)]

    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."
    elapsed = round(time.perf_counter() - t0, 4)

    # 세션 파일명 체크(문서명 리스트)
    check_files: List[str] = []
    try:
        for h in hits_sorted:
            did = h.get("doc_id")
            if did:
                check_files.append(f"{did}.pdf")
            else:
                p = Path(h.get("path", ""))
                if str(p):
                    check_files.append(p.with_suffix(".pdf").name)
    except Exception:
        pass

    return {
        "elapsed_sec": elapsed,
        "settings_used": {"model": model_key, "searchType": search_type},
        "hits": [
            {
                "score": float(h["score"]),
                "path": h["path"],
                "chunk_idx": int(h["chunk_idx"]),
                "task_type": h["task_type"],
                "security_level": int(h["security_level"]),
                "doc_id": h.get("doc_id"),
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
        "check_file": sorted(list(set(check_files))),
        "sid": sid,
        "collection": coll,
    }
# --- add: delete_test_files_by_names ---
async def delete_test_files_by_names(sid: str, file_names: List[str], task_type: Optional[str] = None):
    meta = get_test_session(sid)
    if not meta:
        return {"deleted": 0, "requested": len(file_names), "error": "invalid sid"}

    client = _client()
    coll = meta.get("collection") or _session_collection_name(sid)
    if coll not in client.list_collections():
        return {"deleted": 0, "requested": len(file_names), "error": "collection not found"}

    # 검증
    task_filter = ""
    if task_type:
        if task_type not in TASK_TYPES:
            return {"deleted": 0, "requested": len(file_names), "error": f"invalid taskType: {task_type}"}
        task_filter = f" && task_type == '{task_type}'"

    deleted_total = 0
    per_file: dict[str, int] = {}

    for name in (file_names or []):
        stem = Path(name).stem
        try:
            base_id, _ = _parse_doc_version(stem)
        except Exception:
            base_id = stem
        try:
            filt = f"doc_id == '{base_id}'{task_filter}"
            client.delete(collection_name=coll, filter=filt)
            deleted_total += 1
            per_file[name] = per_file.get(name, 0) + 1
        except Exception:
            logger.exception("[test-delete] failed: %s", name)
            per_file[name] = per_file.get(name, 0)

    try:
        client.flush(coll)
    except Exception:
        pass
    try:
        client.release_collection(collection_name=coll)
    except Exception:
        pass
    try:
        client.load_collection(collection_name=coll)
    except Exception:
        pass

    return {"deleted": deleted_total, "requested": len(file_names), "taskType": task_type, "perFile": per_file, "sid": sid}
