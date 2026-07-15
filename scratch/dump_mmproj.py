import sys
from gguf import GGUFReader

reader = GGUFReader("models/google_gemma-4-E2B-it-mmproj-BF16.gguf")
print("=== METADATA ===")
for key, field in reader.fields.items():
    print(f"{key}: {field.parts[-1] if field.parts else field.value}")

print("\n=== TENSORS ===")
for tensor in reader.tensors:
    print(f"{tensor.name}: {tensor.tensor_type} {tensor.shape}")
