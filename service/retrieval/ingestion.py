from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
from utils.model_load import get_or_load_embedder_async

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedDocument:
    """Normalized ingestion payload (file or dict)."""

    doc_id: str
    version: int
    sec_map: Dict[str, int]
    chunks: List[Dict[str, Any]]
    rel_path: str
    source_path: str
    filename: str
    metadata_records: List[Dict[str, Any]]


async def ingest_common(
    *,
    inputs: Sequence[Union[str, Path, Dict[str, Any]]],
    collection_name: str,
    task_types: Sequence[str],
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

    inputs_seq = list(inputs)
    if not inputs_seq:
        return {"error": "No inputs provided"}

    eff_model_key = settings["embedding_key"]
    tok, model, device = await get_or_load_embedder_async(eff_model_key)
    emb_dim = int(hf_embed_text(tok, model, device, "probe").shape[0])

    client = get_milvus_client()
    ensure_collection_and_index(
        client, emb_dim=emb_dim, metric="IP", collection_name=collection_name
    )

    max_tokens = int(settings.get("chunkSize", 512))
    overlap = int(settings.get("overlap", 64))

    total_inserted_chunks = 0
    processed_files_info = []
    tasks = [str(t) for t in task_types]

    for item in inputs_seq:
        prepared: Optional[PreparedDocument]
        if isinstance(item, (str, Path)):
            prepared = _prepare_file_input(
                Path(item),
                tasks=tasks,
                override_level_map=override_level_map,
                security_level_config=security_level_config,
                doc_id_generator=doc_id_generator,
                doc_id_version_parser=doc_id_version_parser,
                max_tokens=max_tokens,
                overlap=overlap,
                metadata_extras=metadata_extras,
            )
        elif isinstance(item, dict):
            prepared = _prepare_dict_input(
                item,
                tasks=tasks,
                override_level_map=override_level_map,
                upsert_metadata=upsert_metadata,
            )
        else:
            prepared = None

        if not prepared:
            continue

        doc_id = prepared.doc_id
        ver = prepared.version
        sec_map = prepared.sec_map
        chunk_entries = prepared.chunks
        metadata_records = prepared.metadata_records
        rel_path_str = prepared.rel_path
        source_path = prepared.source_path
        file_name_info = prepared.filename

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


def _prepare_file_input(
    path_input: Union[str, Path],
    *,
    tasks: Sequence[str],
    override_level_map: Optional[Dict[str, int]],
    security_level_config: Optional[Dict[str, Dict[str, Any]]],
    doc_id_generator: Optional[Callable[[Any], str]],
    doc_id_version_parser: Optional[Callable[[str], Tuple[str, int]]],
    max_tokens: int,
    overlap: int,
    metadata_extras: Optional[Dict[str, Any]],
) -> Optional[PreparedDocument]:
    file_path = Path(path_input).resolve()
    if not file_path.is_file():
        logger.warning("[ingest] File not found: %s", file_path)
        return None

    try:
        file_text, table_blocks_all = extract_any(file_path)
    except Exception:
        logger.exception("[ingest] Extraction failed for %s", file_path)
        return None

    sec_map = _resolve_security_map(
        tasks,
        override_level_map=override_level_map,
        security_level_config=security_level_config,
        combined_text=(file_text or ""),
        tables=table_blocks_all or [],
    )

    stem = file_path.stem
    if doc_id_version_parser:
        base_doc_id, ver = doc_id_version_parser(stem)
    else:
        base_doc_id, ver = parse_doc_version(stem)
    doc_id = doc_id_generator(base_doc_id) if doc_id_generator else base_doc_id
    rel_path_str = file_path.as_posix()

    chunk_entries: List[Dict[str, Any]] = []
    metadata_records: List[Dict[str, Any]] = []
    metadata_seen: set[int] = set()

    chunks = chunk_text(file_text or "", max_tokens=max_tokens, overlap=overlap)
    for idx, chunk_text_val in enumerate(chunks):
        parts = split_for_varchar_bytes(chunk_text_val)
        for part in parts:
            chunk_entries.append({"page": 0, "chunk_idx": int(idx), "text": part})
        if idx not in metadata_seen:
            payload = {"path": rel_path_str}
            if metadata_extras:
                payload.update(metadata_extras)
            metadata_records.append(
                {
                    "page": 0,
                    "chunk_index": int(idx),
                    "text": chunk_text_val,
                    "payload": payload,
                }
            )
            metadata_seen.add(idx)

    base_idx = len(chunks)
    for table_idx, table in enumerate(table_blocks_all or []):
        md = (table.get("text") or "").strip()
        if not md:
            continue
        page = int(table.get("page", 0))
        bbox = table.get("bbox") or []
        bbox_str = ",".join(str(x) for x in bbox) if bbox else ""
        table_text = f"[[TABLE page={page} bbox={bbox_str}]]\n{md}"

        parts = split_for_varchar_bytes(table_text)
        for sub_j, part in enumerate(parts):
            chunk_idx = base_idx + table_idx * 1000 + sub_j
            chunk_entries.append({"page": page, "chunk_idx": int(chunk_idx), "text": part})
            if chunk_idx in metadata_seen:
                continue
            payload = {"path": rel_path_str, "table": True}
            if metadata_extras:
                payload.update(metadata_extras)
            metadata_records.append(
                {
                    "page": page,
                    "chunk_index": int(chunk_idx),
                    "text": part,
                    "payload": payload,
                }
            )
            metadata_seen.add(chunk_idx)

    return PreparedDocument(
        doc_id=str(doc_id),
        version=int(ver),
        sec_map=sec_map,
        chunks=chunk_entries,
        rel_path=rel_path_str,
        source_path=str(file_path),
        filename=file_path.name,
        metadata_records=metadata_records,
    )


def _prepare_dict_input(
    item: Dict[str, Any],
    *,
    tasks: Sequence[str],
    override_level_map: Optional[Dict[str, int]],
    upsert_metadata: bool,
) -> Optional[PreparedDocument]:
    doc_id = str(item.get("doc_id") or "").strip()
    if not doc_id:
        return None
    version = int(item.get("version", 0))
    chunk_entries = [dict(entry) for entry in item.get("chunks", []) if entry.get("text")]
    if not chunk_entries:
        return None
    sec_map = item.get("levels") or {}
    if override_level_map:
        sec_map = {t: int(override_level_map.get(t, 1)) for t in tasks}
    elif not sec_map:
        sec_map = {t: 1 for t in tasks}

    metadata_records: List[Dict[str, Any]] = []
    if upsert_metadata and item.get("metadata_records"):
        metadata_records = [dict(record) for record in item["metadata_records"]]

    rel_path = str(item.get("source_path") or "")
    return PreparedDocument(
        doc_id=doc_id,
        version=version,
        sec_map=sec_map,
        chunks=chunk_entries,
        rel_path=rel_path,
        source_path=rel_path,
        filename=str(item.get("filename") or doc_id),
        metadata_records=metadata_records,
    )


def _resolve_security_map(
    tasks: Sequence[str],
    *,
    override_level_map: Optional[Dict[str, int]],
    security_level_config: Optional[Dict[str, Dict[str, Any]]],
    combined_text: str,
    tables: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    if override_level_map:
        return {t: int(override_level_map.get(t, 1)) for t in tasks}
    if security_level_config:
        table_text = "\n\n".join(t.get("text", "") for t in tables)
        payload = combined_text + ("\n\n" + table_text if table_text else "")
        return {
            t: determine_level_for_task(
                payload,
                security_level_config.get(t, {"maxLevel": 1, "levels": {}}),
            )
            for t in tasks
        }
    return {t: 1 for t in tasks}
