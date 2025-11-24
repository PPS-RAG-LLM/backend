from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from repository.documents import bulk_upsert_document_metadata
from service.preprocessing.rag_preprocessing import extract_any
from service.retrieval.common import (
    chunk_text,
    determine_level_for_task,
    extract_insert_ids,
    hf_embed_text,
    parse_doc_version,
    split_for_varchar_bytes,
)
from service.vector_db import ensure_collection_and_index, get_milvus_client
from utils.model_load import _get_or_load_embedder_async

logger = logging.getLogger(__name__)


async def ingest_common(
    *,
    # inputs: can be file paths (str/Path) or pre-processed dicts
    inputs: List[Union[str, Path, Dict[str, Any]]],
    collection_name: str,
    task_types: List[str],
    settings: Dict[str, Any],
    security_level_config: Optional[Dict[str, Dict]] = None,
    override_level_map: Optional[Dict[str, int]] = None,
    doc_id_generator: Optional[Callable[[Any], str]] = None,
    doc_id_version_parser: Optional[Callable[[str], Tuple[str, int]]] = None,
    metadata_extras: Optional[Dict[str, Any]] = None,
    post_ingest_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    batch_callback: Optional[Callable[[List[Dict[str, Any]], str], Any]] = None,
    pre_ingest_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    upsert_metadata: bool = True,  # Set False if re-indexing existing metadata
) -> Dict[str, Any]:
    """
    Common ingestion logic.
    Supports two modes per item in `inputs`:
      1. File path (str/Path): Extract -> Chunk -> Embed -> Insert -> (Optional) Upsert Metadata
      2. Dict: { "doc_id":..., "version":..., "chunks": [{"text":..., "page":..., "chunk_idx":...}], "levels":... }
         -> Embed -> Insert -> (Optional) Upsert Metadata (if upsert_metadata=True)
    """

    if not inputs:
        return {"error": "No inputs provided"}

    eff_model_key = settings["embeddingModel"]
    tok, model, device = await _get_or_load_embedder_async(eff_model_key)
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])

    client = get_milvus_client()
    ensure_collection_and_index(
        client, emb_dim=emb_dim, metric="IP", collection_name=collection_name
    )

    max_tokens = int(settings.get("chunkSize", 512))
    overlap = int(settings.get("overlap", 64))

    total_inserted_chunks = 0
    processed_files_info = []

    tasks = task_types

    for item in inputs:
        # --- Mode 1: File Processing ---
        if isinstance(item, (str, Path)):
            file_path = Path(str(item)).resolve()
            if not file_path.is_file():
                logger.warning(f"[ingest] File not found: {file_path}")
                continue

            # Extraction
            try:
                file_text, table_blocks_all = extract_any(file_path)
            except Exception:
                logger.exception(f"[ingest] Extraction failed for {file_path}")
                continue

            # Security Levels
            sec_map = {}
            if override_level_map:
                sec_map = {t: int(override_level_map.get(t, 1)) for t in tasks}
            elif security_level_config:
                whole_for_level = (file_text or "") + "\n\n" + "\n\n".join(
                    t.get("text", "") for t in (table_blocks_all or [])
                )
                sec_map = {
                    t: determine_level_for_task(
                        whole_for_level,
                        security_level_config.get(t, {"maxLevel": 1, "levels": {}}),
                    )
                    for t in tasks
                }
            else:
                sec_map = {t: 1 for t in tasks}

            # Doc ID
            stem = file_path.stem
            if doc_id_version_parser:
                base_doc_id, ver = doc_id_version_parser(stem)
            else:
                base_doc_id, ver = parse_doc_version(stem)
            
            if doc_id_generator:
                doc_id = doc_id_generator(base_doc_id)
            else:
                doc_id = base_doc_id

            rel_path_str = str(file_path.as_posix())
            
            # Chunking & Preparing
            chunk_entries = []
            metadata_records = []
            metadata_seen = set()

            chunks = chunk_text(file_text or "", max_tokens=max_tokens, overlap=overlap)
            
            # Text
            for idx, chunk_text_val in enumerate(chunks):
                parts = split_for_varchar_bytes(chunk_text_val)
                for part in parts:
                    chunk_entries.append({
                        "page": 0,
                        "chunk_idx": int(idx),
                        "text": part,
                    })
                if idx not in metadata_seen:
                    payload = {"path": rel_path_str}
                    if metadata_extras:
                        payload.update(metadata_extras)
                    metadata_records.append({
                        "page": 0,
                        "chunk_index": int(idx),
                        "text": chunk_text_val,
                        "payload": payload,
                    })
                    metadata_seen.add(idx)

            # Tables
            base_idx = len(chunks)
            for t_i, table in enumerate(table_blocks_all or []):
                md = (table.get("text") or "").strip()
                if not md: continue
                page = int(table.get("page", 0))
                bbox = table.get("bbox") or []
                bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
                table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"
                
                parts = split_for_varchar_bytes(table_text)
                for sub_j, part in enumerate(parts):
                    chunk_idx = base_idx + t_i * 1000 + sub_j
                    chunk_entries.append({
                        "page": page,
                        "chunk_idx": int(chunk_idx),
                        "text": part,
                    })
                    if chunk_idx not in metadata_seen:
                        payload = {"path": rel_path_str, "table": True}
                        if metadata_extras:
                            payload.update(metadata_extras)
                        metadata_records.append({
                            "page": page,
                            "chunk_index": int(chunk_idx),
                            "text": part,
                            "payload": payload,
                        })
                        metadata_seen.add(chunk_idx)
            
            source_path = str(file_path)
            file_name_info = file_path.name

        # --- Mode 2: Pre-processed Dict ---
        elif isinstance(item, dict):
            # item keys: doc_id, version, chunks(list), levels(dict), source_path(opt), filename(opt)
            doc_id = item["doc_id"]
            ver = item.get("version", 0)
            chunk_entries = item["chunks"]  # Expected to be pre-split or small enough
            sec_map = item.get("levels") or {}
            if override_level_map:
                sec_map = {t: int(override_level_map.get(t, 1)) for t in tasks}
            elif not sec_map:
                sec_map = {t: 1 for t in tasks}
            
            metadata_records = [] # Assuming metadata already exists for re-indexing
            if upsert_metadata and item.get("metadata_records"):
                metadata_records = item["metadata_records"]

            rel_path_str = item.get("source_path", "")
            source_path = item.get("source_path", "")
            file_name_info = item.get("filename", doc_id)

        else:
            continue

        # Common: Pre-ingest callback (e.g., for creating document records)
        if pre_ingest_callback:
            processed_info_pre = {
                "file": file_name_info,
                "doc_id": doc_id,
                "version": int(ver),
                "levels": sec_map,
                "chunks": 0,
                "source_path": source_path,
            }
            try:
                pre_ingest_callback(processed_info_pre)
            except Exception:
                logger.exception(f"Pre-ingest callback failed for doc_id={doc_id}")

        # Common: Delete existing vectors in Milvus
        try:
            client.delete(
                collection_name=collection_name,
                filter=f"doc_id == '{doc_id}' && version <= {int(ver)}",
            )
        except Exception:
            pass

        # Common: Upsert Metadata (if needed)
        if upsert_metadata and metadata_records:
            try:
                bulk_upsert_document_metadata(doc_id=doc_id, records=metadata_records)
            except Exception:
                logger.exception(f"Failed to upsert metadata for doc_id={doc_id}")

        # Common: Embedding & Insert
        doc_total_chunks = 0
        for t in tasks:
            lvl = int(sec_map.get(t, 1))
            batch = []
            batch_meta = []
            
            local_cnt = 0
            doc_vector_records_for_task = []

            def flush_batch():
                nonlocal batch, batch_meta, local_cnt
                if not batch: return
                try:
                    res = client.insert(collection_name=collection_name, data=batch)
                    ids = extract_insert_ids(res)
                    if batch_callback:
                        for vec_id, meta in zip(ids or [], batch_meta):
                            doc_vector_records_for_task.append({
                                "vector_id": vec_id,
                                "page": meta.get("page", 0),
                                "chunk_index": meta.get("chunk_idx", 0),
                                "task_type": t,
                            })
                    local_cnt += len(batch)
                except Exception:
                    logger.exception(f"[ingest] Insert failed for doc_id={doc_id}")
                finally:
                    batch = []
                    batch_meta = []

            for entry in chunk_entries:
                part = entry.get("text", "")
                if not part: continue
                
                # Safety check for length
                if len(part.encode("utf-8")) > 32768:
                    part = part.encode("utf-8")[:32768].decode("utf-8", errors="ignore")

                vec = hf_embed_text(tok, model, device, part, max_len=max_tokens)
                
                batch.append({
                    "embedding": vec.tolist(),
                    "path": rel_path_str, # or empty string
                    "chunk_idx": int(entry.get("chunk_idx", 0)),
                    "task_type": t,
                    "security_level": lvl,
                    "doc_id": str(doc_id),
                    "version": int(ver),
                    "page": int(entry.get("page", 0)),
                    "workspace_id": 0,
                    "text": part,
                })
                batch_meta.append({"page": entry.get("page", 0), "chunk_idx": int(entry.get("chunk_idx", 0))})
                
                if len(batch) >= 128:
                    flush_batch()
            
            flush_batch()
            doc_total_chunks += local_cnt
            
            if batch_callback and doc_vector_records_for_task:
                batch_callback(doc_vector_records_for_task, doc_id)

        total_inserted_chunks += doc_total_chunks
        
        processed_info = {
            "file": file_name_info,
            "doc_id": doc_id,
            "version": int(ver),
            "levels": sec_map,
            "chunks": doc_total_chunks,
            "source_path": source_path,
        }
        processed_files_info.append(processed_info)
        if post_ingest_callback:
            post_ingest_callback(processed_info)

    try:
        client.flush(collection_name)
    except Exception:
        pass
    ensure_collection_and_index(client, emb_dim=emb_dim, metric="IP", collection_name=collection_name)

    return {
        "message": "Ingestion complete",
        "inserted_chunks": total_inserted_chunks,
        "processed": processed_files_info,
    }
