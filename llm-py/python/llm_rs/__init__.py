"""llm-rs: hardware-agnostic, model-agnostic LLM inference, from Python.

Quickstart::

    from llm_rs import LLM, SamplingParams

    llm = LLM(model="./models/model.gguf")
    outputs = llm.generate(
        ["Hello, how are you?", "What is the capital of France?"],
        SamplingParams(temperature=0.7, max_tokens=64),
    )
    for out in outputs:
        print(out.text)

The hardware backend (CUDA / Metal / CPU) is chosen automatically at load
time by the same runtime `HardwareProfile` dispatch `llm-cli` uses — there is
no device argument here, and none is needed.
"""

from ._llm_rs_native import LLM, RequestOutput, SamplingParams

__all__ = ["LLM", "SamplingParams", "RequestOutput"]
__version__ = "0.1.0"
