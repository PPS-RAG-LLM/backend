"""
python /home/work/CoreIQ/backend/scripts/download_embeddingmodel.py
"""


from huggingface_hub import snapshot_download
import os

# 모델 저장 기본 경로
base_dir = "/home/work/CoreIQ/backend/service/admin/resources/model"

# 다운로드 대상 모델과 저장 폴더명 매핑
models = {
    "Qwen/Qwen3-Embedding-0.6B": "embedding_qwen3_0_6b",
    "BAAI/bge-m3": "embedding_bge_m3",
    "Qwen/Qwen3-Embedding-4B": "embedding_qwen3_4b",
}

def download_models():
    for model_id, folder_name in models.items():
        save_path = os.path.join(base_dir, folder_name)
        print(f"Downloading {model_id} → {save_path}")
        snapshot_download(
            repo_id=model_id,
            local_dir=save_path,
            local_dir_use_symlinks=False
        )
        print(f"✅ Completed: {model_id}")

if __name__ == "__main__":
    download_models()