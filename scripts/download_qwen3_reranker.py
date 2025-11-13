# == scripts/download_qwen3_reranker.py ==
"""
사용법:
  python scripts/download_qwen3_reranker.py --hf_token ""   # 필요 시
  # 저장 경로는 자동으로 ./storage/rerank_model/Qwen3-Reranker-0.6B 로 지정됨
"""
from __future__ import annotations
import os
from typing import Iterable, Optional
from huggingface_hub import snapshot_download, login


def hf_download(
    model_id: str,
    local_dir: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    include: Optional[Iterable[str]] = None,
    local_files_only: bool = False,
    use_symlinks: bool = False,
) -> str:
    if token:
        login(token=token, add_to_git_credential=False)

    os.makedirs(local_dir, exist_ok=True)

    return snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=use_symlinks,
        allow_patterns=list(include) if include else None,
        resume_download=True,
        local_files_only=local_files_only,
    )


def download_qwen3_reranker(
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> dict[str, str]:
    """
    저장 경로: ./storage/rerank_model/Qwen3-Reranker-0.6B
    (리포지토리 루트 기준; 본 스크립트가 scripts/ 에 있다고 가정)
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dst_root = os.path.join(repo_root, "storage", "rerank_model")
    plan = {
        "model_id": "Qwen/Qwen3-Reranker-0.6B",
        "dst": os.path.join(dst_root, "Qwen3-Reranker-0.6B"),
        "include": [
            "*.safetensors",
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.*",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
        ],
    }

    out: dict[str, str] = {}
    out[plan["model_id"]] = hf_download(
        model_id=plan["model_id"],
        local_dir=plan["dst"],
        token=token,
        include=plan["include"],
        local_files_only=local_files_only,
    )
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", default=None, help="Hugging Face 토큰(필요 시)")
    parser.add_argument("--offline", action="store_true", help="캐시만 사용(오프라인)")
    args = parser.parse_args()

    paths = download_qwen3_reranker(
        token=args.hf_token,
        local_files_only=args.offline,
    )
    for mid, p in paths.items():
        print(f"{mid} -> {p}")
