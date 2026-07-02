import os
import sys
import time
from huggingface_hub import hf_hub_download

dest_dir = "/home/mukundan/learning/llm"
os.makedirs(dest_dir, exist_ok=True)

models = [
    {
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
    },
    {
        "repo_id": "ggml-org/SmolLM3-3B-GGUF",
        "filename": "SmolLM3-Q4_K_M.gguf",
    }
]

for m in models:
    repo = m["repo_id"]
    fname = m["filename"]
    target_path = os.path.join(dest_dir, fname)
    if os.path.exists(target_path):
        print(f"File {target_path} already exists. Skipping download.")
        continue

    print(f"Starting download of {fname} from {repo}...")
    start_time = time.time()
    try:
        path = hf_hub_download(
            repo_id=repo,
            filename=fname,
            local_dir=dest_dir,
        )
        duration = time.time() - start_time
        print(f"Successfully downloaded {fname} to {path} in {duration:.1f}s.")
    except Exception as e:
        print(f"Failed to download {fname}: {e}")
        sys.exit(1)

print("All downloads finished successfully!")
