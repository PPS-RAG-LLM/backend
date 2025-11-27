# /home/work/CoreIQ/backend/service/admin/manage_test_LLM.py
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Optional, Dict, Any, List, Tuple
import uuid
from config import config as app_config
from repository.llm_eval_runs import delete_past_eval_runs, find_reusable_run, insert_llm_eval_run
from repository.llm_models import repo_find_best_mapped_model, repo_find_fallback_model
from repository.llm_test_session import get_test_session_by_sid, insert_test_session
from repository.documents import (
    delete_documents_by_type_and_ids,
    insert_document_vectors,
    upsert_document,
    fetch_metadata_by_vector_ids,
)
from repository.prompt_templates.common import repo_find_default_template
from service.retrieval.admin_search import RAGSearchRequest
from storage.db_models import DocumentType
from service.retrieval.common import hf_embed_text
from service.retrieval.ingestion import ingest_common
from service.retrieval.pipeline import DEFAULT_OUTPUT_FIELDS, build_dense_hits, build_rerank_payload
from service.retrieval.reranker import rerank_snippets
from service.vector_db import ensure_collection_and_index, get_milvus_client, run_dense_search, run_hybrid_search

logger = logging.getLogger(__name__)


# ===== 경로/세션 고정값 =====
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
_RETRIEVAL_CFG: Dict[str, Any] = app_config.get("retrieval", {}) or {}
_RETRIEVAL_PATHS: Dict[str, str] = _RETRIEVAL_CFG.get("paths", {}) or {}
_MILVUS_CFG: Dict[str, Any] = _RETRIEVAL_CFG.get("milvus", {}) or {}

LLM_TEST_COLLECTION = _MILVUS_CFG.get("LLM_TEST", "llm_test_collection")

TASK_TYPES = tuple(_RETRIEVAL_CFG.get("task_types") or ("doc_gen", "summary", "qna"))
SUPPORTED_EXTS = set(_RETRIEVAL_CFG.get("supported_extensions"))
_val_rel = _RETRIEVAL_PATHS.get("val_session_root", "storage/val_data") # TODO: MinIO로 마이그레이션
VAL_SESSION_ROOT = (PROJECT_ROOT / _val_rel).resolve()
VAL_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
LLM_TEST_DOC_TYPE = DocumentType.LLM_TEST.value

def _session_doc_id(sid: str, base: str) -> str:
    return f"{sid}:{base}"


def _max_sec_level(sec_map: Dict[str, int]) -> int:
    vals = [int(v) for v in sec_map.values() if v]
    return max(vals or [1])


def _record_llm_test_document(
    *,
    doc_id: str,
    filename: str,
    sid: str,
    rel_text_path: str,
    source_path: str,
    sec_map: Dict[str, int],
    version: int,
) -> None:
    payload = {
        "security_levels": sec_map,
        "version": int(version),
        "session_id": sid,
        "saved_files": {"text": rel_text_path, "source": source_path},
        "source_ext": Path(filename).suffix,
        "updated_at": now_kst_string(),
    }
    upsert_document(
        doc_id=doc_id,
        doc_type=LLM_TEST_DOC_TYPE,
        filename=filename,
        source_path=source_path,
        security_level=_max_sec_level(sec_map),
        payload=payload,
    )
# ===== 외부 의존 (기존 모듈 재사용) =====
from service.admin.manage_admin_LLM import (
    _lookup_model_by_name,
    _fill_template,
    _fetch_prompt_full,
    _norm_category,
    _simple_generate,
)

from service.admin.manage_vator_DB import (
    get_security_level_rules_all,         # (sid, dir) 생성
    parse_doc_version,
)
from utils.model_load import get_or_load_embedder_async, get_vector_settings
from utils.time import now_kst_string

# ===== (옵션) 모델 스트리머 (가능 시 스트림, 실패 시 _simple_generate 폴백) =====
try:
    from utils.llms.huggingface.qwen import stream_chat as _qwen_stream
except Exception:
    _qwen_stream = None

try:
    from utils.llms.huggingface.gpt_oss_20b import stream_chat as _gptoss_stream
except Exception:
    _gptoss_stream = None

from service.admin.manage_admin_LLM import (
    _db_get_model_path as _admin_db_get_model_path,
)

def _select_stream_backend(model_name: str):
    row = _lookup_model_by_name(model_name)
    prov = (row["provider"] if row and "provider" in row.keys() else "") or ""
    name = (model_name or "")
    key = (prov + " " + name).lower()
    if "qwen" in key and _qwen_stream:
        return _qwen_stream
    if ("gpt-oss" in key or "gpt_oss" in key) and _gptoss_stream:
        return _gptoss_stream
    return None

def _infer_answer(prompt_text: str, model_name: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    try:
        model_dir = _admin_db_get_model_path(model_name) 
        backend = _select_stream_backend(model_name)
        if backend and model_dir:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text},
            ]
            chunks: List[str] = []
            for token in backend(messages, model_path=model_dir, temperature=temperature, max_new_tokens=max_tokens):
                chunks.append(token)
            out = "".join(chunks).strip()
            if out:
                return out
    except Exception:
        logger.exception("stream backend inference failed; fallback to _simple_generate")
    try:
        return _simple_generate(prompt_text, model_name, max_tokens=max_tokens, temperature=temperature)
    except Exception:
        logger.exception("_simple_generate failed; returning stub text")
        return "⚠️ 로컬 모델이 로드되지 않아 샘플 응답을 반환합니다. (테스트 전용)"

# ===== 유틸 =====
def _canon_pdf_list(pdf_list: List[str]) -> List[str]:
    names = []
    for f in pdf_list or []:
        name = Path(str(f)).name.strip()
        if name:
            names.append(name)
    # 순서보존 중복제거
    seen = set()
    out = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    # 소문자 정렬 (표준화)
    out = sorted(out, key=lambda s: s.lower())
    return out

def _token_set(s: str) -> set[str]:
    import re
    if not s:
        return set()
    toks = re.findall(r"[0-9A-Za-z가-힣]+", s.lower())
    return set(toks)

def _acc_overlap(a: str, b: str) -> float:
    A, B = _token_set(a), _token_set(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return round(100.0 * inter / max(1, union), 2)

def _ensure_shared_session() -> str:
    """
    DB에서 가장 최근 테스트 세션을 조회하여 사용한다.
    - 없으면 create_test_session()으로 신규 생성
    """
    # 1) DB에서 최신 세션 조회 (가장 최근에 만든 것)
    with get_session() as session:
        # ID 역순(최신순)으로 1개 조회
        row = session.query(LlmTestSession).order_by(LlmTestSession.id.desc()).first()
        if row:
            return row.sid

    # 2) 없으면 신규 생성
    meta = create_test_session()
    if isinstance(meta, dict):
        return meta.get("sid")
    
    # 3) 방어 코드 (거의 발생 안 함)
    import uuid
    return f"shared-{uuid.uuid4().hex[:8]}"

# ===== 파일 관리 =====
def list_shared_files() -> Dict[str, Any]:
    files = []
    for p in sorted(VAL_SESSION_ROOT.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            files.append(p.name)
    return {"success": True, "files": files, "dir": str(VAL_SESSION_ROOT)}

async def upload_shared_files(mem_files: List[tuple[str, bytes]]) -> Dict[str, Any]:
    """
    mem_files: [(filename, bytes), ...]
    - VAL_DIR에 파일 저장(동일 파일명 존재 시 덮어씌움)
    - 공유 세션(sid)에 인제스트
    """
    saved: List[str] = []
    for name, data in mem_files:
        name = Path(name).name.strip()
        if not name:
            continue
        dst = VAL_SESSION_ROOT / name
        dst.write_bytes(data)
        saved.append(str(dst))

    if not saved:
        return {"success": False, "error": "no valid files"}

    sid = _ensure_shared_session()

    # 인제스트 (doc_gen/summary/qna 모두 검색 가능하도록 task_types=None)
    res = await ingest_test_pdfs(sid, saved, task_types=None)
    return {
        "success": True,
        "sid": sid,
        "saved": [Path(p).name for p in saved],
        "ingest": res,
    }

async def delete_shared_files(file_names: List[str]) -> Dict[str, Any]:
    names = [Path(n).name.strip() for n in (file_names or []) if str(n).strip()]
    if not names:
        return {"success": False, "error": "fileNames empty"}

    # 디스크 삭제
    removed: List[str] = []
    for n in names:
        f = VAL_SESSION_ROOT / n
        try:
            if f.is_file():
                f.unlink()
                removed.append(n)
        except Exception:
            logger.exception("failed to remove file: %s", n)

    sid = _ensure_shared_session()
    # 인덱스에서도 제거
    res = await delete_test_files_by_names(sid, file_names=removed, task_type=None)
    return {"success": True, "sid": sid, "removed": removed, "index": res}

# ===== 모델 선택 (기본 템플릿/매핑 기반) =====
def _select_model_for_task(category: str, subcategory: Optional[str]) -> Optional[str]:
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    # 1) default 템플릿 찾기
    tmpl = repo_find_default_template(cat, sub)
    if not tmpl:
        # 2) 카테고리 활성 모델 (Fallback)
        return repo_find_fallback_model(cat)
    prompt_id, _ = tmpl
    # 3) 해당 템플릿 매핑 중 최고 rouge 모델 찾기
    model_name = repo_find_best_mapped_model(prompt_id)
    return model_name

# ===== 핵심: 공유 세션 전체에서 평가 실행 =====
async def ensure_eval_on_shared_session(
    *,
    category: str,
    subcategory: Optional[str],
    prompt_id: int,
    model_name: Optional[str],
    user_prompt: Optional[str],
    top_k: int = 5,
    user_level: int = 1,
    search_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - 공유 세션(유일) 전체에서 RAG 검색 → LLM 생성 → llm_eval_runs 저장
    - 동일키(카테고리/서브카테고리/프롬프트ID/모델명/**user_prompt/pdf_list**) 완전 일치 시 DB 결과를 재사용(모델 미실행)
    - rag_refs에는 RAG 히트의 '실제 출처'를 저장 (예: milvus://<sid>/<doc_id>)
    """
    cat = _norm_category(category)
    sub = (subcategory or "").strip().lower() or None
    user_prompt = (user_prompt or "").strip()

    # 현재 공유 세션 및 파일 목록(정규화)
    sid = _ensure_shared_session()
    current_files = [p.name for p in VAL_SESSION_ROOT.glob("*") if p.is_file() and not p.name.startswith(".")]
    pdf_list = _canon_pdf_list(current_files)
    pdf_json = json.dumps(pdf_list, ensure_ascii=False)

    old = find_reusable_run(
        category=cat,
        subcategory=sub,
        prompt_id=int(prompt_id),
        model_name=(model_name or ""),
        user_prompt=user_prompt,
        pdf_json=pdf_json
    )
    if old:
        try:
            prev_rag_refs = json.loads(old["rag_refs"] or "[]")
        except Exception:
            prev_rag_refs = []
        try:
            prev_pdf_list = json.loads(old["pdf_list"] or "[]")
        except Exception:
            prev_pdf_list = []
        return {
            "success": True,
            "skipped": True,
            "reason": "reuse previous answer (all keys incl. userPrompt & pdf_list matched)",
            "runId": int(old["id"]),
            "category": cat,
            "subcategory": sub,
            "modelName": (model_name or ""),
            "promptId": int(prompt_id),
            "answer": old["answer_text"],
            "acc": old["acc_score"],
            "ragRefs": prev_rag_refs,
            "pdfList": prev_pdf_list,
            "sid": sid,
            "createdAt": old["created_at"],
        }

    # ===== 템플릿 구성 =====
    tmpl, _ = _fetch_prompt_full(prompt_id)
    td = {k: tmpl[k] for k in tmpl.keys()} if not isinstance(tmpl, dict) else tmpl
    system_raw = (td.get("content") or td.get("system_prompt") or "").strip()
    user_raw   = (td.get("sub_content") or td.get("user_prompt") or "").strip()
    tmpl_name  = (td.get("name") or td.get("template_name") or "untitled").strip()

    system_prompt_text = _fill_template(system_raw, {})
    user_prompt_text_from_tmpl = _fill_template(user_raw, {})

    if not model_name:
        model_name = _select_model_for_task(cat, (sub or tmpl_name))
        if not model_name:
            return {"success": False, "error": "모델을 찾을 수 없습니다. 기본/활성 모델을 지정하세요."}

    merged_user = (user_prompt or user_prompt_text_from_tmpl)
    base_prompt_text = (system_prompt_text + ("\n" + merged_user if merged_user else "")).strip()

    # ===== RAG 검색 (공유 세션 전체) =====
    task_for_rag = cat if cat in ("doc_gen", "summary", "qna") else "qna"
    req = RAGSearchRequest(
        query=(user_prompt or tmpl_name or "검색"),
        top_k=top_k,
        user_level=user_level,
        task_type=task_for_rag,
        model=None,
    )
    rag_res = await search_documents_test(req, sid=sid, search_type_override=search_type)
    hits = rag_res.get("hits", []) if isinstance(rag_res, dict) else []

    # rag_refs: 실제 출처를 저장
    rag_refs: List[str] = []
    seen = set()
    for h in hits:
        doc_id = h.get("doc_id") or h.get("document_id") or ""
        if not doc_id:
            path = h.get("path") or ""
            doc_id = Path(path).stem if path else ""
        if not doc_id:
            continue
        key = f"milvus://{sid}/{doc_id}"
        if key in seen:
            continue
        seen.add(key)
        rag_refs.append(key)
    rag_json = json.dumps(rag_refs, ensure_ascii=False)

    # 컨텍스트 결합
    context = "\n---\n".join([h.get("snippet") or "" for h in hits if (h.get("snippet"))]).strip()
    full_prompt = base_prompt_text
    if context:
        full_prompt = f"{base_prompt_text}\n\n[CONTEXT]\n{context}"

    # ===== LLM 생성 =====
    answer = _infer_answer(full_prompt, model_name, max_tokens=512, temperature=0.7)

    # ===== 점수: RAG 컨텍스트 기반 + 프롬프트 겹침 혼합 =====
    try:
        from rouge import Rouge
        _r = Rouge()
        sc = _r.get_scores(answer or "", context or "", avg=True)
        rouge_ctx = (sc.get("rouge-l", {}).get("f", 0.0) or 0.0) * 100.0
    except Exception:
        rouge_ctx = _acc_overlap(context, answer)
    overlap_prompt = _acc_overlap(base_prompt_text, answer)
    acc = round(0.7 * rouge_ctx + 0.3 * overlap_prompt, 2)

    # ===== 저장 =====
    mrow = _lookup_model_by_name(model_name)
    llm_id = int(mrow["id"]) if mrow else None

    run_id = insert_llm_eval_run(
        llm_id=llm_id,
        prompt_id=int(prompt_id),
        category=cat,
        subcategory=(sub or tmpl_name),
        model_name=model_name,
        prompt_text=full_prompt,
        user_prompt=user_prompt,
        rag_json=rag_json,
        answer=answer,
        acc=acc,
        meta_json=json.dumps({"source": "ensure-on-shared", "sid": sid, "top_k": top_k, "user_level": user_level}, ensure_ascii=False),
        pdf_json=pdf_json
    )

    return {
        "success": True,
        "created": True,
        "runId": run_id,
        "category": cat,
        "subcategory": (sub or tmpl_name),
        "modelName": model_name,
        "promptId": int(prompt_id),
        "answer": answer,
        "acc": acc,
        "ragRefs": rag_refs,
        "pdfList": pdf_list,
        "sid": sid,
    }


# ===== 과거 답 삭제 =====
def delete_past_runs(
    *,
    run_id: Optional[int] = None,
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    prompt_id: Optional[int] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    - run_id가 오면 해당 1건 삭제
    - 아니면 (category, subcategory, prompt_id, model_name) 조합으로 삭제
    - 최소 1가지 조건은 필요
    """
    if not run_id and not any([category, subcategory, prompt_id, model_name]):
        return {"success": False, "error": "no criteria"}

    try:
        deleted_count = delete_past_eval_runs(
            run_id=run_id,
            category=category,
            subcategory=subcategory,
            prompt_id=prompt_id,
            model_name=model_name
        )
        return {
            "success": True, 
            "deleted": deleted_count, 
            "by": {
                "category": category, "subcategory": subcategory, "promptId": prompt_id, "modelName": model_name
            }
        }
    except Exception:
        logger.exception("delete_past_runs failed")
        return {"success": False, "error": "delete_past_runs failed"}


# --- add: search_documents_test ---
async def search_documents_test(req: RAGSearchRequest, sid: str, search_type_override: Optional[str] = None, rerank_top_n: Optional[int] = None) -> Dict:
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

    tok, model, device = await get_or_load_embedder_async(model_key)
    q_emb = hf_embed_text(tok, model, device, req.query)

    client = get_milvus_client()
    coll = meta.get("collection") 
    ensure_collection_and_index(client, emb_dim=len(q_emb), metric="IP", collection_name=coll)
    if coll not in client.list_collections():
        return {"error": "세션 컬렉션이 없습니다. 먼저 인제스트 하세요."}

    embedding_candidates = int(req.top_k)  # 임베딩에서 찾을 후보 개수
    final_results = int(rerank_top_n) if rerank_top_n is not None else 5  # 최종 반환 개수
    candidate = max(embedding_candidates, final_results * 2)
    filter_expr = f"task_type == '{req.task_type}' && security_level <= {int(req.user_level)}"

    if search_type == "vector":
        raw_results = run_dense_search(
            client,
            collection_name=coll,
            query_vector=q_emb.tolist(),
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
    else:
        raw_results = run_hybrid_search(
            client,
            collection_name=coll,
            query_vector=q_emb.tolist(),
            query_text=req.query,
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=DEFAULT_OUTPUT_FIELDS,
        )
    hits_raw = build_dense_hits(raw_results)

    vector_ids = [str(h["vector_id"]) for h in hits_raw if h.get("vector_id")]
    meta_map = fetch_metadata_by_vector_ids(vector_ids)
    for i, hit in enumerate(hits_raw):
        vid = str(hit.get("vector_id") or "")
        meta = meta_map.get(vid)
        logger.debug(f"[Search '{i}'] vector_id: {vid} meta: {(meta["text"][:50] + "...")}")
        if meta:
            hit["doc_id"] = hit.get("doc_id") or meta.get("doc_id")
            hit["chunk_idx"] = meta.get("chunk_index")
            hit["snippet"] = meta.get("text")
       
            
    rerank_candidates = build_rerank_payload(hits_raw)

    if rerank_candidates:
        reranked = rerank_snippets(rerank_candidates, query=req.query, top_n=final_results)
        hits_sorted = []
        for res in reranked:
            original = res.metadata or {}
            hits_sorted.append(
                {
                    "score": float(res.score),
                    "doc_id": original.get("doc_id"),
                    "path": original.get("path"),
                    "vector_id": original.get("vector_id"),
                    "chunk_idx": int(original.get("chunk_idx", 0)),
                    "task_type": original.get("task_type"),
                    "security_level": int(original.get("security_level", 1)),
                    "page": int(original.get("page", 0)),
                    "snippet": res.text,
                }
            )
    else:
        hits_sorted = sorted(
            hits_raw,
            key=lambda x: x.get("score_fused", x.get("score_vec", x.get("score_sparse", 0.0))),
            reverse=True,
        )[:final_results]

    # 리랭크 결과 로그 출력 (테스트 세션)
    if hits_sorted:
        top_hit = hits_sorted[0]
        logger.info(f"✨ [Rerank-Test] 완료! 최고 점수: {top_hit.get('score', 0):.4f}")
        logger.info(f"🏆 [Rerank-Test] 최고 스니펫 (doc_id: {top_hit.get('doc_id', 'unknown')}): {top_hit.get('snippet', '')[:100]}...")

    context = "\n---\n".join(h["snippet"] for h in hits_sorted if h.get("snippet"))
    prompt = f"사용자 질의: {req.query}\n:\n{context}\n\n위 내용을 바탕으로 응답을 생성해 주세요."
    elapsed = round(time.perf_counter() - t0, 4)


    return {
        "elapsed_sec": elapsed,
        "settings_used": {"model": model_key, "searchType": search_type},
        "hits": [
            {
                "score": float(h.get("score", h.get("score_fused", h.get("score_vec", 0.0)))),
                "doc_id": h.get("doc_id"),
                "path": h.get("path", ""),
                "vector_id": h["vector_id"],
                "chunk_idx": int(h["chunk_idx"]),
                "task_type": h["task_type"],
                "security_level": int(h["security_level"]),
                "page": int(h["page"]),
                "snippet": h["snippet"],
            }
            for h in hits_sorted
        ],
        "prompt": prompt,
        "sid": sid,
        "collection": coll,
    }


def _serialize_session(row: LlmTestSession) -> Dict[str, Any]:
    return {
        "sid": row.sid,
        "dir": row.directory,
        "collection": row.collection,
        "createdAt": row.created_at.isoformat() if row.created_at else None,
    }

from typing import Any, Optional, Dict
import uuid
from utils import logger
from utils.database import get_session
from sqlalchemy import select
from storage.db_models import LlmTestSession

logger = logger(__name__)


def create_test_session() -> Dict:
    sid = uuid.uuid4().hex[:12]
    # DB 에는 경로 대신 'virtual' 또는 빈 문자열
    # 혹은 나중에 원본 폴더(VAL_DIR)를 참조하기 위한 용도로 VAL_DIR 경로를 저장해도 됨
    # 여기서는 단순히 str(VAL_DIR)을 넣어 "이 세션은 이 폴더를 쓴다"는 의미로 남기거나
    # 아예 필요 없다면 빈 문자열로 처리 ㅇ
    common_dir = str(VAL_SESSION_ROOT) 
    obj = insert_test_session(sid, common_dir, LLM_TEST_COLLECTION)
    return _serialize_session(obj)

def get_test_session(sid: str) -> Optional[Dict]:
    row = get_test_session_by_sid(sid)
    return _serialize_session(row)

# --- 
# add: delete_test_files_by_names ---
async def delete_test_files_by_names(sid: str, file_names: List[str], task_type: Optional[str] = None):
    meta = get_test_session(sid)
    if not meta:
        return {"deleted": 0, "requested": len(file_names), "error": "invalid sid"}

    client = get_milvus_client()
    coll = meta.get("collection") 
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

    doc_ids_to_remove: set[str] = set()

    for name in (file_names or []):
        stem = Path(name).stem
        try:
            base_id, _ = parse_doc_version(stem)
        except Exception:
            base_id = stem
        doc_key = _session_doc_id(sid, base_id)
        doc_ids_to_remove.add(doc_key)
        try:
            filt = f"doc_id == '{doc_key}'{task_filter}"
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

    if doc_ids_to_remove:
        try:
            delete_documents_by_type_and_ids(LLM_TEST_DOC_TYPE, list(doc_ids_to_remove))
        except Exception:
            logger.exception("[test-delete] failed to remove document metadata")

    return {"deleted": deleted_total, "requested": len(file_names), "taskType": task_type, "perFile": per_file, "sid": sid}
    
# --- add: ingest_test_pdfs ---
async def ingest_test_pdfs(sid: str, pdf_paths: List[str], task_types: Optional[List[str]] = None):
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}

    coll = meta.get("collection")
    settings = get_vector_settings()
    all_rules = get_security_level_rules_all()
    tasks = task_types or list(TASK_TYPES)

    def _doc_id_gen(base: str) -> str:
        return _session_doc_id(sid, base)
        
    def _post_ingest(info: Dict):
        # info has file, doc_id, version, levels, chunks, source_path
        _record_llm_test_document(
            doc_id=info["doc_id"],
            filename=info["file"],
            sid=sid,
            rel_text_path=str(Path(info["source_path"]).as_posix()), 
            source_path=info["source_path"],
            sec_map=info["levels"],
            version=info["version"]
        )

    def _batch_callback(records: List[Dict[str, Any]], doc_id: str):
        if not records:
            return
        try:
            insert_document_vectors(
                doc_id=doc_id,
                collection=coll,
                embedding_version=str(settings["embeddingModel"]),
                vectors=records,
            )
        except Exception:
            logger.exception(f"document_vectors 기록 실패(doc_id={doc_id})")

    res = await ingest_common(
        inputs=pdf_paths,
        collection_name=coll,
        task_types=tasks,
        settings=settings,
        security_level_config=all_rules,
        doc_id_generator=_doc_id_gen,
        pre_ingest_callback=_post_ingest,
        post_ingest_callback=_post_ingest,
        batch_callback=_batch_callback,
    )
    
    return {"message": "세션 인제스트 완료", "sid": sid, "inserted_chunks": res["inserted_chunks"]}
