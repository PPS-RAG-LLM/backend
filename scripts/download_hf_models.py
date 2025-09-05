"""
# 다운로드

python scripts/download_hf_models.py --base_dir /storage/model --hf_token ""

"""

# download_hf_models.py
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
    """
    Hugging Face 모델 체크포인트를 로컬에 스냅샷로 다운로드.
    - resume_download=True 로 부분 다운로드/중단 재개 지원
    - allow_patterns 로 불필요 파일 제외 가능
    - local_files_only=True 로 오프라인/캐시 우선 사용

    Returns: 실제 저장된 로컬 디렉터리 경로
    """
    if token:
        # git-credential에 저장하지 않음(서버 환경 안전성)
        login(token=token, add_to_git_credential=False)

    os.makedirs(local_dir, exist_ok=True)

    # 고속 전송(선택): pip install hf_transfer 후 아래 환경변수 활성화
    # os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        local_dir_use_symlinks=use_symlinks,  # 기본 False: 실파일 복사
        allow_patterns=list(include) if include else None,
        resume_download=True,
        local_files_only=local_files_only,
    )
    return path

def download_default_two(
    base_dir: str,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> dict[str, str]:
    """
    본 요청의 두 모델을 내려받음.
    include 패턴은 안전하게 필수 파일만 지정(용량 절감).
    """
    plans = [
        {
            "model_id": "openai/gpt-oss-20b",
            "dst": os.path.join(base_dir, "gpt-oss-20b"),
            # gpt-oss 파일 구성: *.safetensors, *config.json, tokenizer.*, chat_template 등
            "include": [
                "*.safetensors",
                "*.json",
                "tokenizer.*",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "chat_template*",
            ],
        },
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct-1M",
            "dst": os.path.join(base_dir, "Qwen2.5-7B-Instruct-1M"),
            "include": [
                "*.safetensors",
                "*.json",
                "tokenizer.*",
                "vocab.json",
                "merges.txt",
            ],
        },
    ]
    out = {}
    for p in plans:
        out[p["model_id"]] = hf_download(
            model_id=p["model_id"],
            local_dir=p["dst"],
            token=token,
            include=p["include"],
            local_files_only=local_files_only,
        )
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="모델 저장 루트 디렉터리")
    parser.add_argument("--hf_token", default=None, help="Hugging Face 토큰(필요 시)")
    parser.add_argument("--offline", action="store_true", help="캐시만 사용(오프라인)")
    args = parser.parse_args()

    paths = download_default_two(
        base_dir=args.base_dir,
        token=args.hf_token,
        local_files_only=args.offline,
    )
    for mid, p in paths.items():
        print(f"{mid} -> {p}")