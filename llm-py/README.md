# llm-rs (Python bindings)

Hardware-agnostic, model-agnostic LLM inference — from Python, in the style of vLLM's
`LLM`/`SamplingParams` API.

```python
from llm_rs import LLM, SamplingParams

llm = LLM(model="./models/model.gguf")
outputs = llm.generate(
    ["Hello, how are you?", "What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=64),
)
for out in outputs:
    print(out.text)
```

The hardware backend (CUDA / Metal / CPU) is auto-detected at load time by the same
`HardwareProfile` runtime dispatch `llm-cli` uses — there is no device argument, and
none is needed. Multimodal models (vision/audio encoders) are detected the same way;
`llm.has_vision_encoder`/`llm.has_audio_encoder` report what was found.

## Install (from source)

```bash
pip install maturin
cd llm-py
maturin develop --release   # builds and installs into the active venv
```

## API

- `LLM(model, tokenizer_path=None, explicit_dequantize=False, use_vram_embeddings=False, block_pool_size=1024)`
  — loads a GGUF file or an HF-style safetensors directory. Raises a clear `FileNotFoundError`/
  `RuntimeError` on a bad path or load failure — never silently substitutes a placeholder model.
- `llm.generate(prompts: list[str], sampling_params: SamplingParams | None = None) -> list[RequestOutput]`
  — batches all prompts through the same continuous-batching scheduler `llm serve` uses.
- `SamplingParams(temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.1, max_tokens=256)`
  — validated at construction time (bad values raise `ValueError` immediately, not silently
  clamped).
- `RequestOutput.text`, `.token_ids`, `.finish_reason`, `.prefill_tokens_per_sec`, `.decode_tokens_per_sec`.

## Notes

- This binds directly to the `llm-core`/`llm-scheduler` Rust crates (not through the C
  `llm-ffi` layer) — each `LLM` instance owns its own Tokio runtime, so nothing here requires
  an `asyncio` event loop on the Python side.
- `llm.generate()` releases the GIL while it runs, so other Python threads aren't blocked for
  the duration of a (potentially long) generation call.
- **Tokenization is handled entirely in Rust** (via `llm-core`'s `LlmTokenizer`, itself backed
  by HuggingFace's `tokenizers` crate) — you do **not** need the Python `tokenizers` package
  installed, and `LLM.generate()` never calls back into Python for encode/decode.
- Current tokenizer support is HF "fast tokenizer" `tokenizer.json` files only (this matches
  what `llm-cli`/`llm serve` support today). SentencePiece `tokenizer.model`-only repos
  (no `tokenizer.json` present) aren't supported yet — `LLM(...)` raises a clear
  `FileNotFoundError` telling you to pass `tokenizer_path=` explicitly rather than failing
  silently or guessing.
