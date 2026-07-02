import os
from huggingface_hub import hf_hub_download

dest_dir = "/home/mukundan/learning/llm"

print("Downloading SmolLM3 tokenizer.json...")
try:
    path = hf_hub_download(
        repo_id="HuggingFaceTB/SmolLM3-3B",
        filename="tokenizer.json",
        local_dir=dest_dir,
    )
    os.rename(path, os.path.join(dest_dir, "smollm3_tokenizer.json"))
    print("SmolLM3 tokenizer.json downloaded successfully!")
except Exception as e:
    print(f"Error downloading SmolLM3 tokenizer: {e}")
