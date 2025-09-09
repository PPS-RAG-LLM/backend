from __future__ import annotations

def free_torch_memory(sync: bool = True) -> None:
    """
    안전한 메모리 정리:
    - Python GC 수집
    - CUDA 사용 가능 시, 동기화 후 캐시 비우기
    """
    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                if sync:
                    torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        # torch 미설치 또는 예외 시 무시
        pass