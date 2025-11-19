"""LLM 기반 리랭커 유틸리티."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from service.retrieval.adapters.base import RetrievalResult
from utils import logger

LOGGER = logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RERANK_MODEL_PATH = PROJECT_ROOT / "storage" / "rerank_model" / "Qwen3-Reranker-0.6B"

_RERANK_CACHE: dict[str, Tuple[Any, Any, torch.device, int, int]] = {}
_ACTIVE_RERANK_KEY: str | None = None


def _load_reranker() -> Tuple[Any, Any, torch.device, int, int]:
    """로컬 Qwen 리랭커 모델 로딩."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    need_files = [
        RERANK_MODEL_PATH / "config.json",
        RERANK_MODEL_PATH / "tokenizer.json",
    ]
    missing = [str(path) for path in need_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Reranker model files missing: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(RERANK_MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(RERANK_MODEL_PATH),
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device).eval()

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    return tokenizer, model, device, token_true_id, token_false_id


def load_reranker_model(cache_key: str = "qwen3_reranker_0.6b"):
    """캐시를 이용해 리랭커 모델을 반환."""
    global _ACTIVE_RERANK_KEY
    if cache_key == _ACTIVE_RERANK_KEY and cache_key in _RERANK_CACHE:
        return _RERANK_CACHE[cache_key]
    try:
        model = _load_reranker()
    except FileNotFoundError:
        LOGGER.warning("리랭커 모델이 존재하지 않습니다. (경로: %s)", RERANK_MODEL_PATH)
        return None
    _RERANK_CACHE.clear()
    _RERANK_CACHE[cache_key] = model
    _ACTIVE_RERANK_KEY = cache_key
    return model


def _format_instruction(instruction: str, query: str, doc: str) -> str:
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query,
        doc=doc,
    )


def _normalize_snippet(snippet: Any) -> Tuple[str, float]:
    """RetrievalResult 또는 dict에서 텍스트/점수 추출."""
    if isinstance(snippet, RetrievalResult):
        return snippet.text, float(snippet.score)
    if is_dataclass(snippet):
        data = asdict(snippet)
    elif isinstance(snippet, dict):
        data = snippet
    else:
        raise TypeError("Unsupported snippet type for rerank")
    return str(data.get("text", "")), float(data.get("score", 0.0))


def _compute_scores(
    tokenizer,
    model,
    device,
    token_true_id: int,
    token_false_id: int,
    pairs: Sequence[str],
) -> List[float]:
    """허깅페이스 예제 기반으로 yes/no 확률 계산."""
    if not pairs:
        return []

    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the "
        "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    inputs = tokenizer(
        list(pairs),
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=8192 - len(prefix_tokens) - len(suffix_tokens),
    )

    for i, seq in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + seq + suffix_tokens

    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :]
    true_vector = logits[:, token_true_id]
    false_vector = logits[:, token_false_id]
    stacked = torch.stack([false_vector, true_vector], dim=1)
    stacked = torch.nn.functional.log_softmax(stacked, dim=1)
    return stacked[:, 1].exp().tolist()


def rerank_snippets(
    snippets: Iterable[Any],
    query: str,
    *,
    instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
    top_n: int = 5,
) -> List[RetrievalResult]:
    """
    스니펫 목록을 리랭크하여 상위 top_n을 반환.

    snippets 항목은 RetrievalResult 또는 dict 형태를 지원한다.
    """
    normalized: List[Tuple[Any, str, float]] = []
    for item in snippets:
        text, score = _normalize_snippet(item)
        if not text.strip():
            continue
        normalized.append((item, text, score))
    if not normalized:
        return []

    model_bundle = load_reranker_model()
    if model_bundle is None:
        LOGGER.warning("리랭커 모델을 찾을 수 없어 기존 점수를 사용합니다.")
        sorted_items = sorted(normalized, key=lambda x: x[2], reverse=True)
        return [
            _to_result(item[0], item[2]) for item in sorted_items[: top_n or len(sorted_items)]
        ]

    tokenizer, model, device, token_true_id, token_false_id = model_bundle
    pairs = [_format_instruction(instruction, query, text) for (_, text, _) in normalized]

    scores: List[float] = []
    batch_size = 16
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        try:
            scores.extend(
                _compute_scores(tokenizer, model, device, token_true_id, token_false_id, chunk)
            )
        except Exception as exc:  # pragma: no cover - 안전장치
            LOGGER.warning("리랭크 배치 실패: %s", exc)
            scores.extend([0.0] * len(chunk))

    reranked = []
    for (item, _text, _orig_score), rerank_score in zip(normalized, scores):
        reranked.append((item, float(rerank_score)))
    reranked.sort(key=lambda x: x[1], reverse=True)

    return [
        _to_result(item, rerank_score)
        for item, rerank_score in reranked[: top_n or len(reranked)]
    ]


def _to_result(snippet: Any, score: float) -> RetrievalResult:
    """다양한 입력 타입을 RetrievalResult로 변환."""
    if isinstance(snippet, RetrievalResult):
        snippet.score = score
        return snippet

    if is_dataclass(snippet):
        data = asdict(snippet)
    elif isinstance(snippet, dict):
        data = dict(snippet)
    else:
        raise TypeError("Unsupported snippet type")

    return RetrievalResult(
        doc_id=data.get("doc_id"),
        title=str(data.get("title") or data.get("doc_id") or "snippet"),
        text=str(data.get("text") or ""),
        score=score,
        source=str(data.get("source") or "unknown"),
        chunk_index=data.get("chunk_index"),
        page=data.get("page"),
        metadata=data.get("metadata") or {},
    )

