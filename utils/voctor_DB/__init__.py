from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional
from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer  # type: ignore

# Simple module-level logger
logger = logging.getLogger(__name__)


# ------------------------------
# Paths / Defaults
# ------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # .../backend
DEFAULT_MODEL_ROOT = (PROJECT_ROOT / "storage" / "embedding-models").resolve()


# ------------------------------
# Model Resolution Helpers
# ------------------------------
def _list_candidate_model_dirs(model_root_dir: Path) -> List[Path]:
    if not model_root_dir.exists():
        return []
    return [p.resolve() for p in model_root_dir.iterdir() if p.is_dir()]


def _aliases(p: Path) -> List[str]:
    name = p.name.lower()
    res = [name]
    if name.startswith("embedding_"):
        res.append(name[len("embedding_"):])
    return res


def resolve_model_dir(model_key: str, model_root_dir: Optional[Path] = None) -> Path:
    """
    Resolve a local model directory from a model key.
    Priority:
      1) exact match or alias match
      2) substring match
      3) fallback to a directory named 'qwen3_0_6b' if present
    """
    key = (model_key or "").lower().strip()
    root = (model_root_dir or DEFAULT_MODEL_ROOT).resolve()
    cands = _list_candidate_model_dirs(root)

    # exact/alias
    for p in cands:
        if key and key in _aliases(p):
            return p
    # partial
    for p in cands:
        if key and key in p.name.lower():
            return p
    # fallback
    for p in cands:
        if "qwen3_0_6b" in p.name.lower():
            return p
    raise FileNotFoundError(f"No local model directory found for key '{model_key}' under {root}")


# ------------------------------
# Embedding Model Loading
# ------------------------------
def _ensure_required_files(model_dir: Path) -> None:
    need = [model_dir/"tokenizer_config.json", model_dir/"tokenizer.json", model_dir/"config.json"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        logger.error(f"[Embedding] Missing required files in {model_dir}: {missing}")
        raise FileNotFoundError(f"Embedding model files missing: {model_dir}")


@lru_cache(maxsize=8)
def _load_from_dir(dir_str: str, device_str: str) -> Tuple[any, any, torch.device]:
    model_dir = Path(dir_str).resolve()
    _ensure_required_files(model_dir)
    device = torch.device(device_str)
    logger.info(f"[Embedding] Loading model from {model_dir} (device={device_str})")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if device_str == "cuda" else torch.float32,
    ).to(device).eval()
    logger.info(f"[Embedding] Loaded model from {model_dir}")
    return tok, model, device


def load_embedding_model(model_key: str, model_root_dir: Optional[Path] = None) -> Tuple[any, any, torch.device]:
    """
    Load tokenizer/model/device from local directory only, using an LRU cache.
    Raises FileNotFoundError on missing files or directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = resolve_model_dir(model_key, model_root_dir=model_root_dir)
    return _load_from_dir(str(model_dir), "cuda" if device.type == "cuda" else "cpu")


# ------------------------------
# Text Embedding
# ------------------------------
def _mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


def embed_text(tokenizer, model, device, text: str, max_len: int = 512) -> List[float]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="longest",
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outs = model(**inputs)
    vec = _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
    return vec.tolist()


# ------------------------------
# Chunk/Overlap Manager
# ------------------------------
def get_vector_defaults_from_config() -> Tuple[int, int]:
    try:
        from config import config  # lazy import to avoid cycles
        conf = config.get("vector_defaults") or {}
        chunk_size = int(conf.get("chunk_size", 512))
        overlap = int(conf.get("overlap", 64))
        return chunk_size, overlap
    except Exception:
        return 512, 64


def chunk_text_with_overlap(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    """
    Chunk text by words with overlap. If chunk_size/overlap are None, use config vector_defaults.
    Ensures 0 <= overlap < chunk_size, and returns non-empty chunks only.
    """
    if chunk_size is None or overlap is None:
        cs, ov = get_vector_defaults_from_config()
        chunk_size = chunk_size or cs
        overlap = overlap or ov

    chunk_size = int(max(1, chunk_size))
    overlap = int(max(0, min(overlap, max(0, chunk_size - 1))))

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        start += max(1, chunk_size - overlap)
    return chunks


__all__ = [
    "resolve_model_dir",
    "load_embedding_model",
    "embed_text",
    "chunk_text_with_overlap",
    "get_vector_defaults_from_config",
]



