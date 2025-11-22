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
from repository.llm_test_session import get_test_session_by_sid, insert_test_session
from service.preprocessing.rag_preprocessing import ext, extract_any
from service.retrieval.common import hf_embed_text, chunk_text
from service.retrieval.pipeline.milvus_pipeline import build_dense_hits, build_rerank_payload, load_snippet_from_store
from service.retrieval.reranker import rerank_snippets
from service.vector_db import ensure_collection_and_index, get_milvus_client, run_dense_search, run_hybrid_search
from functools import partial
logger = logging.getLogger(__name__)

# ===== 경로/세션 고정값 =====
BASE_DIR = Path(__file__).resolve().parent  # .../backend/service/admin
PROJECT_ROOT = BASE_DIR.parent.parent  # .../backend
_RETRIEVAL_CFG: Dict[str, Any] = app_config.get("retrieval", {}) or {}
_RETRIEVAL_PATHS: Dict[str, str] = _RETRIEVAL_CFG.get("paths", {}) or {}
_MILVUS_CFG: Dict[str, Any] = _RETRIEVAL_CFG.get("milvus", {}) or {}

LLM_TEST_COLLECTION = _MILVUS_CFG.get("LLM_TEST", "llm_test_collection")

def _cfg_path(key: str, fallback: str) -> Path:
    value = _RETRIEVAL_PATHS.get(key, fallback)
    return (PROJECT_ROOT / Path(value)).resolve()

_BACKEND_ROOT = Path(__file__).resolve().parents[2]     # .../backend
VAL_DIR = _BACKEND_ROOT / "storage" / "val_data"        # TODO: DB로 마이그레이션
VAL_DIR.mkdir(parents=True, exist_ok=True)              # TODO: DB로 마이그레이션
TASK_TYPES = tuple(_RETRIEVAL_CFG.get("task_types") or ("doc_gen", "summary", "qna"))
SUPPORTED_EXTS = set(_RETRIEVAL_CFG.get("supported_extensions"))

EXTRACTED_TEXT_DIR = _cfg_path("extracted_text_dir", "storage/extracted_texts")
_SHARED_META = VAL_DIR / ".shared_session.json"         # TODO: DB로 마이그레이션
VAL_SESSION_ROOT = _cfg_path("val_session_root", "storage/val_data")
# ===== 외부 의존 (기존 모듈 재사용) =====
from service.admin.manage_admin_LLM import (
    _connect,
    _lookup_model_by_name,
    _fill_template,
    _fetch_prompt_full,
    _norm_category,
    _simple_generate,
)

from service.admin.manage_vator_DB import (
    determine_level_for_task,
    get_security_level_rules_all,         # (sid, dir) 생성
    get_vector_settings,            # sid -> meta
    RAGSearchRequest,
    parse_doc_version,
    split_for_varchar_bytes,
)
from utils.model_load import _get_or_load_embedder_async

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
    _resolve_model_fs_path as _admin_resolve_model_fs_path,
)

def _resolve_local_model_dir_for_infer(model_name: str) -> Optional[str]:
    p = _admin_db_get_model_path(model_name) or _admin_resolve_model_fs_path(model_name)
    if not p:
        return None
    p = p.replace("\\", "/")
    if p.startswith("./"):
        return str((_BACKEND_ROOT / p.lstrip("./")).resolve())
    return p

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
        model_dir = _resolve_local_model_dir_for_infer(model_name)
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

# ===== 공유 세션 보장 =====
def _load_shared_sid_from_disk() -> Optional[str]:
    try:
        if _SHARED_META.is_file():
            data = json.loads(_SHARED_META.read_text(encoding="utf-8"))
            sid = data.get("sid")
            if sid and isinstance(sid, str):
                return sid
    except Exception:
        logger.exception("failed to load shared sid")
    return None

def _save_shared_sid_to_disk(sid: str) -> None:
    try:
        _SHARED_META.write_text(json.dumps({"sid": sid}, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.exception("failed to save shared sid")

def _ensure_shared_session() -> str:
    """
    공유 세션 SID를 파일로 유지한다.
    - 파일에 없으면 create_test_session()으로 신규 생성 후 저장
    - 재시작해도 동일 SID 사용
    """
    sid = _load_shared_sid_from_disk()
    if sid:
        meta = get_test_session(sid)
        if meta:
            return sid
        # 세션 메타가 사라졌으면 재생성
    meta = create_test_session()
    if isinstance(meta, dict):
        sid = meta.get("sid") or meta.get("id") or meta.get("session_id")
    elif isinstance(meta, (list, tuple)) and len(meta) >= 1:
        sid = meta[0]
    else:
        sid = None
    if not sid:
        import uuid
        sid = f"shared-{uuid.uuid4().hex[:8]}"
    _save_shared_sid_to_disk(sid)
    return sid

# ===== 파일 관리 =====
def list_shared_files() -> Dict[str, Any]:
    files = []
    for p in sorted(VAL_DIR.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            files.append(p.name)
    return {"success": True, "files": files, "dir": str(VAL_DIR)}

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
        dst = VAL_DIR / name
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
        f = VAL_DIR / n
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
    conn = _connect()
    cur = conn.cursor()
    try:
        # 1) default 템플릿 찾기
        if sub:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND lower(name)=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
                """,
                (cat, sub),
            )
        else:
            cur.execute(
                """
                SELECT id, name FROM system_prompt_template
                 WHERE category=? AND ifnull(is_default,0)=1 AND ifnull(is_active,1)=1
                 ORDER BY id DESC LIMIT 1
                """,
                (cat,),
            )
        tmpl = cur.fetchone()
        if not tmpl:
            # 2) 카테고리 활성 모델
            cur.execute(
                """
                SELECT name FROM llm_models
                 WHERE is_active=1 AND (category=? OR category='all')
                 ORDER BY trained_at DESC, id DESC LIMIT 1
                """,
                (cat,),
            )
            r = cur.fetchone()
            return r[0] if r else None

        prompt_id = int(tmpl["id"])
        # 3) 해당 템플릿 매핑 중 최고 rouge
        cur.execute(
            """
            SELECT llm_id FROM llm_prompt_mapping
             WHERE prompt_id=?
             ORDER BY IFNULL(rouge_score,-1) DESC, llm_id DESC
             LIMIT 1
            """,
            (prompt_id,),
        )
        mp = cur.fetchone()
        if not mp:
            return None
        cur.execute("SELECT name FROM llm_models WHERE id=?", (mp["llm_id"],))
        r = cur.fetchone()
        return r[0] if r else None
    finally:
        conn.close()

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
    current_files = [p.name for p in VAL_DIR.glob("*") if p.is_file() and not p.name.startswith(".")]
    pdf_list = _canon_pdf_list(current_files)
    pdf_json = json.dumps(pdf_list, ensure_ascii=False)

    conn = _connect()
    cur = conn.cursor()
    try:
        # ===== 재사용 조건에 user_prompt, pdf_list까지 포함 =====
        cur.execute(
            """
            SELECT id, answer_text, acc_score, rag_refs, pdf_list, created_at, user_prompt, prompt_text
              FROM llm_eval_runs
             WHERE category=?
               AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')
               AND prompt_id=?
               AND model_name=?
               AND IFNULL(user_prompt,'') = IFNULL(?, '')
               AND IFNULL(pdf_list,'[]') = ?
             ORDER BY id DESC
             LIMIT 1
            """,
            (cat, sub, int(prompt_id), (model_name or ""), user_prompt, pdf_json),
        )
        old = cur.fetchone()
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

        cur.execute(
            """
            INSERT INTO llm_eval_runs(
              mapping_id, llm_id, prompt_id, category, subcategory, model_name,
              prompt_text, user_prompt, rag_refs, answer_text, acc_score, meta, pdf_list
            )
            VALUES(NULL,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                llm_id,
                int(prompt_id),
                cat,
                (sub or tmpl_name),
                model_name,
                full_prompt,
                user_prompt,
                rag_json,
                answer,
                acc,
                json.dumps({"source": "ensure-on-shared", "sid": sid, "top_k": top_k, "user_level": user_level}, ensure_ascii=False),
                pdf_json,
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)

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
    except Exception:
        logger.exception("ensure_eval_on_shared_session failed")
        return {"success": False, "error": "ensure_eval_on_shared_session failed"}
    finally:
        conn.close()

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

    conn = _connect()
    cur = conn.cursor()
    try:
        if run_id:
            cur.execute("DELETE FROM llm_eval_runs WHERE id=?", (int(run_id),))
            conn.commit()
            return {"success": True, "deleted": cur.rowcount, "by": {"runId": run_id}}

        cat = _norm_category(category) if category else None
        sub = (subcategory or "").strip().lower() if subcategory is not None else None

        sql = "DELETE FROM llm_eval_runs WHERE 1=1"
        params: List[Any] = []
        if cat is not None:
            sql += " AND category=?"
            params.append(cat)
        if sub is not None:
            sql += " AND IFNULL(LOWER(subcategory),'') = IFNULL(LOWER(?),'')"
            params.append(sub)
        if prompt_id is not None:
            sql += " AND prompt_id=?"
            params.append(int(prompt_id))
        if model_name is not None:
            sql += " AND model_name=?"
            params.append(model_name)

        cur.execute(sql, tuple(params))
        conn.commit()
        return {"success": True, "deleted": cur.rowcount, "by": {"category": cat, "subcategory": sub, "promptId": prompt_id, "modelName": model_name}}
    except Exception:
        logger.exception("delete_past_runs failed")
        return {"success": False, "error": "delete_past_runs failed"}
    finally:
        conn.close()



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

    tok, model, device = await _get_or_load_embedder_async(model_key)
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
    snippet_loader = partial(
        load_snippet_from_store,
        EXTRACTED_TEXT_DIR,
        max_tokens=int(settings["chunkSize"]),
        overlap=int(settings["overlap"]),
    )
    output_fields = ("path", "chunk_idx", "task_type", "security_level", "doc_id", "text")

    if search_type == "vector":
        res_dense = run_dense_search(
            client,
            collection_name=coll,
            query_vector=q_emb.tolist(),
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        hits_raw = build_dense_hits(res_dense, snippet_loader=snippet_loader)
    else:
        res_hybrid = run_hybrid_search(
            client,
            collection_name=coll,
            query_vector=q_emb.tolist(),
            query_text=req.query,
            limit=candidate,
            filter_expr=filter_expr,
            output_fields=output_fields,
        )
        hits_raw = build_dense_hits(res_hybrid, snippet_loader=snippet_loader)

    rerank_candidates = build_rerank_payload(hits_raw)

    if rerank_candidates:
        reranked = rerank_snippets(rerank_candidates, query=req.query, top_n=final_results)
        hits_sorted = []
        for res in reranked:
            original = res.metadata or {}
            hits_sorted.append(
                {
                    "score": float(res.score),
                    "path": original.get("path"),
                    "chunk_idx": int(original.get("chunk_idx", 0)),
                    "task_type": original.get("task_type"),
                    "security_level": int(original.get("security_level", 1)),
                    "doc_id": original.get("doc_id"),
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
    sess_dir = (VAL_SESSION_ROOT / sid).resolve()
    sess_dir.mkdir(parents=True, exist_ok=True)
    obj = insert_test_session(sid, sess_dir, LLM_TEST_COLLECTION)
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

    for name in (file_names or []):
        stem = Path(name).stem
        try:
            base_id, _ = parse_doc_version(stem)
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
    
# --- add: ingest_test_pdfs ---
async def ingest_test_pdfs(sid: str, pdf_paths: List[str], task_types: Optional[List[str]] = None):
    meta = get_test_session(sid)
    if not meta:
        return {"error": "invalid sid"}

    settings = get_vector_settings()
    eff_model_key = settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])

    client = get_milvus_client()
    coll = meta.get("collection")
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=coll)

    MAX_TOKENS, OVERLAP = int(settings["chunkSize"]), int(settings["overlap"])

    all_rules = get_security_level_rules_all()
    sess_txt_root = EXTRACTED_TEXT_DIR / "__sessions__" / sid
    sess_txt_root.mkdir(parents=True, exist_ok=True)

    tasks = task_types or list(TASK_TYPES)
    total = 0

    for p in pdf_paths:
        file_path = Path(str(p))
        if ext(file_path) not in SUPPORTED_EXTS:
            logger.warning(f"[test-ingest] Unsupported file type: {file_path}")
            continue

        try:
            file_text, table_blocks_all = extract_any(file_path)
        except Exception:
            logger.exception("[test-ingest] read failed: %s", p)
            continue

        whole_for_level = file_text + "\n\n" + "\n\n".join(t.get("text","") for t in (table_blocks_all or []))
        sec_map = {t: determine_level_for_task(whole_for_level, all_rules.get(t, {"maxLevel": 1, "levels": {}})) for t in TASK_TYPES}
        max_sec = max(sec_map.values()) if sec_map else 1
        sec_folder = f"securityLevel{int(max_sec)}"

        rel_txt = Path("__sessions__") / sid / sec_folder / file_path.with_suffix(".txt").name
        abs_txt = EXTRACTED_TEXT_DIR / rel_txt
        abs_txt.parent.mkdir(parents=True, exist_ok=True)
        abs_txt.write_text(file_text, encoding="utf-8")

        stem = file_path.stem
        doc_id, ver = parse_doc_version(stem)

        try:
            client.delete(coll, filter=f"doc_id == '{doc_id}' && version <= {int(ver)}")
        except Exception:
            pass

        chunks = chunk_text(file_text, max_tokens=MAX_TOKENS, overlap=OVERLAP)  # pyright: ignore[reportUndefinedVariable]
        batch: List[Dict] = []

        for t in tasks:
            lvl = int(sec_map.get(t, 1))

            # 본문: VARCHAR 안전 분할
            for idx, c in enumerate(chunks):
                for part in split_for_varchar_bytes(c):
                    if len(part.encode("utf-8")) > 32768:
                        part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                    vec = hf_embed_text(tok, model, device, part, max_len=MAX_TOKENS)
                    batch.append({
                        "embedding": vec.tolist(),
                        "path": str(rel_txt.as_posix()),
                        "chunk_idx": int(idx),
                        "task_type": t,
                        "security_level": lvl,
                        "doc_id": str(doc_id),
                        "version": int(ver),
                        "page": 0,
                        "workspace_id": 0,
                        "text": part,
                    })
                    if len(batch) >= 128:
                        client.insert(collection_name=coll, data=batch)
                        total += len(batch)
                        batch = []

            # 표: VARCHAR 안전 분할
            base_idx = len(chunks)
            for t_i, table in enumerate(table_blocks_all or []):
                md = (table.get("text") or "").strip()
                if not md:
                    continue
                page = int(table.get("page", 0))
                bbox = table.get("bbox") or []
                bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"

                for sub_j, part in enumerate(split_for_varchar_bytes(table_text)):
                    if len(part.encode("utf-8")) > 32768:
                        part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")
                    vec = hf_embed_text(tok, model, device, part, max_len=MAX_TOKENS)
                    batch.append({
                        "embedding": vec.tolist(),
                        "path": str(rel_txt.as_posix()),
                        "chunk_idx": int(base_idx + t_i * 1000 + sub_j),
                        "task_type": t,
                        "security_level": lvl,
                        "doc_id": str(doc_id),
                        "version": int(ver),
                        "page": int(page),
                        "workspace_id": 0,
                        "text": part,
                    })
                    if len(batch) >= 128:
                        client.insert(collection_name=coll, data=batch)
                        total += len(batch)
                        batch = []

        if batch:
            client.insert(collection_name=coll, data=batch)
            total += len(batch)

    try:
        client.flush(coll)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=coll)

    return {"message": "세션 인제스트 완료", "sid": sid, "inserted_chunks": total}
