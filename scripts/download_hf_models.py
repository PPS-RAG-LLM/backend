"""
# 다운로드

python scripts/download_hf_models.py --base_dir /storage/model --hf_token ""

사용법:
python scripts/download_hf_models.py --base_dir /storage/model --hf_token ""
# 또는
export HF_TOKEN=""
python scripts/download_hf_models.py --base_dir /storage/model
"""
from __future__ import annotations

import os
import sys
import errno
from typing import Iterable, Optional, Dict

from huggingface_hub import snapshot_download, login


def ensure_writable_dir(path: str) -> None:
    """디렉토리 생성 및 쓰기 가능 여부 사전 점검."""
    os.makedirs(path, exist_ok=True)
    testfile = os.path.join(path, ".write_test.tmp")
    try:
        with open(testfile, "w") as f:
            f.write("ok")
        os.remove(testfile)
    except OSError as e:
        raise RuntimeError(
            f"[권한 오류] '{path}' 에 쓰기 권한이 없습니다. "
            f"sudo chown -R $USER:$USER '{path}' 또는 다른 경로를 사용하세요."
        ) from e


def hf_download(
    model_id: str,
    local_dir: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    include: Optional[Iterable[str]] = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> str:
    """
    Hugging Face 모델을 로컬 디렉터리로 스냅샷 다운로드.
    - huggingface_hub >= 0.24 기준, resume_download는 디폴트 동작이라 인자 제거.
    - local_dir_use_symlinks(Deprecated) 제거.
    """
    # 토큰 우선순위: 인자 > 환경변수(HF_TOKEN)
    if token:
        # git-credential에 저장하지 않음(서버 안전용)
        login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token  # snapshot_download 내부에서도 동일 토큰 사용

    ensure_writable_dir(local_dir)

    # 병렬/재시도시 lock 충돌 방지 팁:
    # 1) 동일 local_dir로 다중 프로세스 동시 실행 금지
    # 2) 필요시 모델별로 서로 다른 local_dir 사용

    path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_dir,
        allow_patterns=list(include) if include else None,
        local_files_only=local_files_only,
        force_download=force_download,  # True면 강제 재다운로드
    )
    return path


def download_default_two(
    base_dir: str,
    token: Optional[str] = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> Dict[str, str]:
    """
    지정된 두 모델 다운로드.
    필요 파일만 include로 한정해 용량/시간 절약.
    """
    plans = [
        {
            "model_id": "openai/gpt-oss-20b",
            "dst": os.path.join(base_dir, "gpt-oss-20b"),
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

    out: Dict[str, str] = {}
    for p in plans:
        out[p["model_id"]] = hf_download(
            model_id=p["model_id"],
            local_dir=p["dst"],
            token=token,
            include=p["include"],
            local_files_only=local_files_only,
            force_download=force_download,
        )
    return out


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True, help="모델 저장 루트 디렉터리")
    parser.add_argument("--hf_token", default=None, help="Hugging Face 토큰(선택)")
    parser.add_argument("--offline", action="store_true", help="캐시만 사용(오프라인)")
    parser.add_argument("--force", action="store_true", help="강제 재다운로드")
    args = parser.parse_args()

    try:
        paths = download_default_two(
            base_dir=args.base_dir,
            token=args.hf_token,
            local_files_only=args.offline,
            force_download=args.force,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    for mid, p in paths.items():
        print(f"{mid} -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
