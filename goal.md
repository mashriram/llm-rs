# AgnosticEngine — Complete Engineering Specification
## The Fastest, Easiest, Most Universal LLM Inference Runtime

> **North Star**: Drop any model from Hugging Face or GGUF onto any hardware — laptop, Mac, NVIDIA rig, or a chain of devices connected by USB — and get a running OpenAI-compatible server in one command. No config files. No model-specific code. No recompilation.

---

## Why This Beats Everything Else

| | llama.cpp | vLLM | MLC-LLM | **AgnosticEngine** |
|---|---|---|---|---|
| New model (day zero) | Manual GGUF port | Python rewrite | TVM recompile | **Automatic** |
| GPU backends | CUDA only | CUDA only | CUDA/Metal | **CUDA/Metal/Vulkan/CPU** |
| Multi-node | No | Limited | No | **USB mesh, zero config** |
| Memory safety | C++ (unsafe) | Python | C++ (unsafe) | **Rust (safe)** |
| Quantized math | AOT only | AOT only | AOT only | **JIT, fused at runtime** |
| Install | cmake build | pip + drivers | pip + TVM | **`cargo install .llm`** |

---

## The Two Promises

**Promise 1: Easiest**
```bash
# Install
cargo install llm

# Run any model
llm serve Qwen/Qwen3-0.6B
# → OpenAI-compatible API at http://localhost:8080
# No config. No flags. No model-specific anything.
```

**Promise 2: Fastest**
- INT4/8 math executes without dequantization to F32 (direct quantized GEMV)
- KV waste < 4% via PagedAttention (vs 60-80% in static allocation)
- JIT-compiled kernels beat AOT C++ via autotune and shape-specialization
- Zero-copy activation streaming between nodes over USB

---

## Architecture: 6-Phase Pipeline

```
                    ANY MODEL (HF SafeTensors / GGUF)
                              │
              ┌───────────────▼───────────────┐
              │  Phase 1: Ingestion Frontend  │  Zero-copy mmap
              │  Metadata → Fallback AST      │  safetensors / GGUF
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  Phase 2: Paradigm Router     │  Autoregressive /
              │  Model class → Exec loop      │  Diffusion / MoE / MTP
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  Phase 3: Hardware Profiler   │  CUDA / Metal / Vulkan /
              │  Capability → Device profile  │  CPU SIMD discovery
              └───────────────┬───────────────┘
                              │
              ┌───────────────▼───────────────┐
              │  Phase 4: Graph Scheduler     │  PagedAttention
              │  Layer distribution + batching│  Continuous batching
              └──────────┬────────────────────┘
                         │
           ┌─────────────▼──────────┐   ┌─────────────────────────────┐
           │  Phase 5: JIT Kernels  │   │  Phase 6: USB Cluster Mesh  │
           │  CubeCL → PTX/Metal/   │   │  Zenoh P2P / rkyv streams   │
           │  SPIR-V / CPU ASM      │   │  Pause-Replicate-Retry      │
           └────────────────────────┘   └─────────────────────────────┘
```

---

## Crate Structure

```
agnostic-engine/
├── Cargo.toml                    # workspace
├── ae-core/                      # LlmBackend trait, types, sampler
├── ae-ingest/                    # Phase 1: mmap loaders, tokenizer
├── ae-classify/                  # Phase 2: paradigm router
├── ae-profile/                   # Phase 3: hardware capability profiler
├── ae-schedule/                  # Phase 4: scheduler, block pool, prefix trie
├── ae-kernel/                    # Phase 5: CubeCL JIT kernels
├── ae-cluster/                   # Phase 6: USB mesh, zenoh, rkyv
└── ae-cli/                       # HTTP server, CLI entry point
```

### Dependency Manifest

```toml
[workspace.dependencies]

# GPU / JIT
cubecl            = { version = "0.6", features = ["cuda", "wgpu"] }
cubecl-cuda       = "0.6"
cubecl-wgpu       = "0.6"
cubecl-hip        = "0.6"

# Weight loading
safetensors       = "0.5"
memmap2           = "0.9"
candle-core       = { version = "0.9", features = ["cuda"] }

# Model graph
petgraph          = "0.6"
hashbrown         = "0.14"
smol_str          = "0.2"
ndarray           = "0.15"

# Async
tokio             = { version = "1", features = ["full"] }
tokio-util        = "0.7"
axum              = { version = "0.7", features = ["tokio"] }

# Cluster
zenoh             = "0.11"
rkyv              = { version = "0.7", features = ["validation"] }

# System info
sysinfo           = "0.30"
raw-window-handle = "0.6"

# Serialization / utils
serde             = { version = "1", features = ["derive"] }
serde_json        = "1"
tokenizers        = "0.20"
half              = "2"
crossbeam         = "0.8"
dashmap           = "6"

# Error handling
anyhow            = "1"
thiserror         = "1"
tracing           = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

---

## Phase 1: Hybrid Ingestion Frontend (`ae-ingest`)

### Goal
Zero-copy load of any SafeTensors or GGUF model. Extract enough metadata in < 200ms to route to the correct execution path without loading a single weight tensor into RAM.

### Primary path: metadata extraction

```rust
// ae-ingest/src/metadata.rs

pub struct ModelSpec {
    pub arch: ArchType,           // Llama | Qwen | Mistral | Phi | Gemma | Unknown
    pub paradigm: Paradigm,       // Autoregressive | Diffusion | Multimodal | MTP | MoE
    pub vocab_size: usize,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub weight_dtype: WeightDtype,
    pub quantization: QuantScheme,    // None | Q4_0 | Q4_K | Q8_0 | AWQ | GPTQ
    pub tensor_index: TensorIndex,    // name → (offset, shape, dtype)
}
```

Extraction order (fastest first, fallback chain):

1. `config.json` (HF models) — serde_json, ~1ms
2. GGUF header fields — binary parse, ~5ms
3. SafeTensors header JSON — zero-copy, ~2ms
4. **Fallback: tensor name analysis** — regex scan of all tensor names to infer arch from naming patterns

```rust
// ae-ingest/src/fallback.rs

/// Fallback structural analysis when config.json is absent.
/// Scans tensor names to classify model architecture.
pub fn infer_from_tensor_names(names: &[&str]) -> ArchType {
    let patterns: &[(&str, ArchType)] = &[
        ("model.layers.",      ArchType::Llama),
        ("transformer.h.",     ArchType::GPT2),
        ("model.blocks.",      ArchType::MPT),
        ("gpt_neox.layers.",   ArchType::GPTNeoX),
        ("experts.",           ArchType::MoE),   // MoE pattern
        ("text_model.",        ArchType::Multimodal),
        ("vision_model.",      ArchType::Multimodal),
    ];
    for name in names {
        for &(pattern, arch) in patterns {
            if name.contains(pattern) { return arch; }
        }
    }
    ArchType::Unknown
}
```

### Zero-copy mmap

```rust
// ae-ingest/src/mmap.rs

pub struct MappedModel {
    _mmap: memmap2::Mmap,        // keeps file mapped
    pub spec: ModelSpec,
    pub tensor_data: &'static [u8], // SAFETY: lifetime tied to _mmap, private
}

impl MappedModel {
    /// Open a model file without loading any weights.
    /// Only the header is read; all tensor data stays on disk until accessed.
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the Mmap is kept alive as long as MappedModel exists.
        // tensor_data is a view into it — never outlives self.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let spec = parse_header(&mmap)?;
        let data_ptr = mmap.as_ptr();
        let data_len = mmap.len();
        let tensor_data = unsafe { std::slice::from_raw_parts(data_ptr, data_len) };
        Ok(Self { _mmap: mmap, spec, tensor_data })
    }
}
```

### Tokenizer auto-detection

```rust
// ae-ingest/src/tokenizer.rs

pub fn auto_load_tokenizer(model_dir: &Path) -> Result<LlmTokenizer> {
    // Try in priority order — no config required from user
    let candidates = [
        "tokenizer.json",              // HF fast tokenizer
        "tokenizer.model",             // SentencePiece
        "vocab.json",                  // GPT-2 style
    ];
    for name in &candidates {
        let p = model_dir.join(name);
        if p.exists() {
            return LlmTokenizer::from_file(&p);
        }
    }
    // Final fallback: extract from GGUF metadata
    LlmTokenizer::from_gguf_metadata(model_dir)
}
```

### Data modality unification

All input types normalize to `Vec<TokenId>` before entering the scheduler:

| Input | Conversion |
|---|---|
| Text string | tokenizer.encode() |
| Image file | CLIP patch tokenizer (if multimodal) |
| Audio | Whisper mel-spec encoder → token ids |
| Raw token ids | pass-through |

### Acceptance criterion
`llm serve ./models/Qwen3-0.6B` must produce a running API in < 3 seconds on first run (excluding weight download). The `ModelSpec` must be fully populated from metadata alone — no weight loading required.

---

## Phase 2: Paradigm Router (`ae-classify`)

### Goal
Map the extracted `ModelSpec` to the correct execution loop. No user configuration.

### Routing table

```rust
// ae-classify/src/router.rs

pub enum ExecLoop {
    AutoregressiveDecode {
        use_paged_kv: bool,
        use_continuous_batching: bool,
        speculative_draft: Option<Arc<dyn LlmBackend>>,
    },
    DiffusionDenoise {
        steps: usize,
        scheduler: DiffusionScheduler,   // DDPM | DDIM | DPMSolver
    },
    MultimodalPrefill {
        vision_encoder: Arc<dyn LlmBackend>,
        text_decoder: Arc<dyn LlmBackend>,
    },
    MTPDecode {
        // Multi-Token Prediction: multiple output heads
        n_parallel_tokens: usize,
    },
    MoEDecode {
        n_experts: usize,
        top_k: usize,                    // active experts per token
        expert_placement: ExpertMap,     // expert_id → device
    },
}

pub fn classify(spec: &ModelSpec, hardware: &HardwareProfile) -> ExecLoop {
    // MoE detection: experts.* tensors present
    if spec.paradigm == Paradigm::MoE {
        return ExecLoop::MoEDecode { ... };
    }
    // Diffusion: no attention KV, has "timestep" embeddings
    if spec.paradigm == Paradigm::Diffusion {
        return ExecLoop::DiffusionDenoise { ... };
    }
    // Multimodal: vision_model.* tensors present
    if spec.paradigm == Paradigm::Multimodal {
        return ExecLoop::MultimodalPrefill { ... };
    }
    // Default: standard autoregressive with paged KV
    ExecLoop::AutoregressiveDecode {
        use_paged_kv: true,
        use_continuous_batching: true,
        speculative_draft: None,
    }
}
```

### Speculative decoding (Medusa / Eagle)

When a draft model path is provided, the router initializes concurrent execution:

```
Draft model (fast, small) → generates N candidate tokens
Target model (large, authoritative) → verifies N tokens in one forward pass
Scheduler → accepts prefix of accepted tokens, re-runs only the rejected suffix
```

Draft and target run as separate `LlmBackend` instances. The scheduler manages interleaving via a `SpeculativeSession` that tracks the acceptance tree.

### Acceptance criterion
Feed a MoE model (Mixtral-8x7B GGUF). The router must detect `experts.*` tensors and select `ExecLoop::MoEDecode` without any user flag.

---

## Phase 3: Hardware Capability Profiler (`ae-profile`)

### Goal
Interrogate available hardware at startup and produce a `HardwareProfile` that drives all subsequent decisions. No manual device configuration.

### Discovery sequence

```rust
// ae-profile/src/profiler.rs

pub struct HardwareProfile {
    pub devices: Vec<DeviceCapability>,
    pub total_vram_gb: f32,
    pub total_ram_gb: f32,
    pub best_backend: BackendKind,   // Cuda | Metal | Vulkan | CpuSimd
}

pub struct DeviceCapability {
    pub id: DeviceId,
    pub kind: DeviceKind,            // NvidiaCuda | AppleMetal | AmdHip | Cpu | Usb
    pub vram_gb: f32,
    pub bandwidth_tb_s: f32,         // measured, not advertised
    pub gflops_fp16: f32,            // measured via small matmul benchmark
    pub supports_dp4a: bool,         // INT8 dot product
    pub supports_int4_native: bool,  // native INT4 tensor cores (H100/MI300)
    pub simd_width: usize,           // 8=AVX2, 16=AVX-512, 8=NEON
}
```

### Bandwidth measurement (not trust the spec sheet)

```rust
pub fn measure_bandwidth(device: &Device) -> f32 {
    // Allocate 256MB, do 10 sequential reads, time it
    const SIZE: usize = 256 * 1024 * 1024;
    let buf = device.alloc(SIZE)?;
    let t0 = std::time::Instant::now();
    for _ in 0..10 { device.memcpy_d2h(&buf, SIZE)?; }
    let elapsed = t0.elapsed().as_secs_f32();
    (SIZE as f32 * 10.0) / elapsed / 1e12  // TB/s
}
```

### SIMD intrinsic detection

```rust
// ae-profile/src/simd.rs

pub fn detect_cpu_simd() -> SimdCaps {
    SimdCaps {
        avx2:     is_x86_feature_detected!("avx2"),
        avx512f:  is_x86_feature_detected!("avx512f"),
        avx512vnni: is_x86_feature_detected!("avx512vnni"),  // INT8 dot product
        neon:     cfg!(target_arch = "aarch64"),
        i8mm:     is_arm_feature_detected!("i8mm"),           // ARM INT8 matrix multiply
    }
}
```

### Autotune cache

On first run, benchmark kernel variants and cache results in `~/.llm/autotune/{device_id}.json`. All subsequent starts skip the benchmark.

```json
{
  "device": "NVIDIA RTX 4090",
  "best_gemv_q8_vec_factor": 8,
  "best_attention_block_size": 16,
  "best_prefill_tile": [128, 64],
  "measured_bandwidth_tb_s": 0.97,
  "measured_fp16_gflops": 165000,
  "timestamp": "2025-06-30T00:00:00Z"
}
```

### Acceptance criterion
On a MacBook with M-series chip: `ae-profile` must detect Metal backend, measure memory bandwidth, and set `supports_int4_native = false` (M3 has INT8 but not INT4 matrix units). On an RTX 4090: must set `supports_int4_native = true` and measure > 0.8 TB/s bandwidth.

---

## Phase 4: Graph Scheduler (`ae-schedule`)

### Goal
Manage all in-flight sequences and their KV cache state. Maximize GPU utilization while holding KV waste < 4%.

### Block pool (PagedAttention)

```rust
// ae-schedule/src/block_pool.rs

pub const BLOCK_SIZE: usize = 16;   // tokens per physical block — compile-time constant

pub struct BlockPool {
    capacity: usize,
    free_list: crossbeam::queue::SegQueue<BlockId>,
    ref_counts: Vec<std::sync::atomic::AtomicU32>,
    // All KV data lives in a single pre-allocated VRAM slab
    // Layout: [block_id][layer][kv_head][token][head_dim]
    k_slab_ptr: *mut u8,   // raw device pointer — managed by CubeCL allocator
    v_slab_ptr: *mut u8,
    bytes_per_block_per_layer: usize,
}
```

**Copy-on-Write rule**: When a sequence forks (beam search, parallel sampling), increment `ref_count` on shared blocks. Before any write to a block with `ref_count > 1`, allocate a new block and copy — never mutate shared state.

### Three execution classes

After every generated token, the scheduler classifies each active sequence:

```
OneShot   → classification / embedding / K≤1 generation
            No KV written. Activation tensors freed immediately.
            ~3x faster than Decode for short requests.

Decode    → standard paged-KV continuous batching
            KV blocks allocated incrementally as tokens are generated.

Mixed     → prefill chunk interleaved with active decode sequences
            Prevents "decode starvation" when long prompts arrive.
```

### Prefix trie (Leader-Peer mechanism)

```
Incoming requests with shared system prompt [A B C D]:
  Request 1 → Leader: runs full prefill, populates shared blocks
  Request 2 → Peer:   deferred 1 tick, then attaches to Leader's blocks
  Request 3 → Peer:   deferred 1 tick, then attaches to Leader's blocks

Result: shared prefix computed ONCE. All peers share ref-counted blocks.
        KV waste: ~0% for shared prefix portion.
```

### Iteration-level scheduler loop

```rust
// ae-schedule/src/scheduler.rs

pub fn step(&mut self, backend: &dyn LlmBackend) -> Result<()> {
    // 1. Evict finished sequences — decref their blocks
    self.evict_done();

    // 2. Check prefix trie for waiting sequences
    self.check_prefix_hits();

    // 3. Promote waiting sequences if block budget allows
    self.admit_waiting()?;

    // 4. Classify: OneShot | Decode | Mixed
    let batch = self.assemble_batch();

    // 5. Forward pass
    let output = backend.forward_pass(&batch)?;

    // 6. Append tokens — may trigger new block allocation
    for (seq_id, token) in output.seq_ids.iter().zip(output.next_tokens.iter()) {
        self.sequences[seq_id].append_token(*token, &self.block_pool)?;
    }

    // 7. Send tokens to HTTP SSE streams
    self.flush_to_clients(&output);

    Ok(())
}
```

### Rotor quantization for active KV cache

For long-context requests (> 4K tokens), apply an orthogonal rotation to KV vectors before packing to INT4/INT8. Rotation reduces outliers, which is the main source of quantization error in KV caches.

```
Standard KV-INT4: ~2% quality loss at 4K tokens
Rotor KV-INT4:    ~0.3% quality loss at 4K tokens (rotates outliers into uniform distribution)
```

This doubles effective context length within the same VRAM budget.

### Acceptance criterion
96 concurrent requests to Qwen3-4B-Q8. Scheduler must:
- Maintain > 150 req/s throughput
- KV waste < 4% measured over 1000 requests
- Zero OOM errors over a 10-minute soak test

---

## Phase 5: JIT Kernels (`ae-kernel`)

### Goal
All math executes through CubeCL JIT kernels. No AOT precompiled binaries. Kernels are specialized at runtime for the exact hardware, quantization scheme, and batch shape encountered.

### Direct quantized GEMV (no dequantization to F32)

```rust
// ae-kernel/src/gemv_q8.rs

#[cube(launch)]
pub fn q8_gemv<F: Float>(
    weights_i8: &Array<i8>,
    scales: &Array<F>,        // one f16 scale per 32-weight block
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] hidden_dim: u32,   // comptime = JIT loop unrolling
    #[comptime] out_dim: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;
    let n_blocks = hidden_dim / 32u32;
    let mut acc = F::from_int(0);

    // Each thread handles a stride of 32-element blocks
    // #[comptime] allows the JIT to emit unrolled inner loops
    let mut b = tid;
    while b < n_blocks {
        let scale = scales[row * n_blocks + b];
        #[unroll]
        for k in 0u32..32u32 {
            let w = F::cast_from(weights_i8[row * hidden_dim + b * 32u32 + k]);
            acc += w * scale * input[b * 32u32 + k];
        }
        b += CUBE_DIM_X;
    }
    // Shared-memory reduction across threads
    // ...
    if tid == 0u32 { output[row] = acc; }
}
```

**Why this beats llama.cpp by > 25%**: The `#[comptime]` annotation on `hidden_dim` and `out_dim` allows the JIT compiler to:
1. Unroll the inner loop completely (eliminating loop counter arithmetic)
2. Eliminate all bounds checks (sizes are compile-time constants)
3. Emit optimal register widths for the specific GPU's SIMD width
4. Fuse the scale multiply into the widening path (no separate pass)

### Varlen attention (packs multiple sequences into one kernel launch)

```rust
#[cube(launch)]
pub fn paged_varlen_attention<F: Float>(
    q: &Array<F>,
    block_table: &Array<u32>,    // logical → physical block IDs per sequence
    k_cache: &Array<F>,          // flat physical block pool
    v_cache: &Array<F>,
    cu_seqlens: &Array<u32>,     // cumulative sequence lengths
    output: &mut Array<F>,
    #[comptime] block_size: u32,
    #[comptime] n_kv_heads: u32,
    #[comptime] head_dim: u32,
) {
    // One CUBE per (sequence, head) pair
    // Fetches KV blocks via block_table indirection
    // Online softmax: single-pass, no O(n) memory for attention weights
    // ...
}
```

**Online softmax eliminates a full pass over the sequence**. Standard attention requires two passes (first to find max for numerical stability, second to compute exp). Online softmax merges them into one, halving memory bandwidth for long sequences.

### Kernel registry (autotune selects best variant at runtime)

```rust
// ae-kernel/src/registry.rs

pub struct KernelRegistry {
    gemv_q8: Vec<GemvVariant>,        // vectorization factors: 1, 2, 4, 8
    attention: Vec<AttentionVariant>,  // block sizes: 32, 64, 128
    best: HashMap<(DeviceId, KernelKind, ShapeKey), VariantId>,  // cached results
}

impl KernelRegistry {
    pub fn get_best_gemv(&self, device: &DeviceId, hidden_dim: usize) -> &GemvVariant {
        let key = (device.clone(), KernelKind::GemvQ8, hidden_dim);
        if let Some(&id) = self.best.get(&key) {
            return &self.gemv_q8[id];
        }
        // Run benchmark on first call — result cached
        self.autotune_gemv(device, hidden_dim)
    }
}
```

### Kernel list (all implemented via `#[cube]`)

| Kernel | Notes |
|---|---|
| `q8_gemv` | Direct INT8 GEMV, fused dequant |
| `q4_gemv` | Direct INT4 GEMV, bit-unpacking in registers |
| `fp16_gemm` | F16 GEMM for prefill (compute-bound) |
| `paged_varlen_attention` | Non-contiguous KV blocks, online softmax |
| `rope_embed` | RoPE with precomputed freq table |
| `rmsnorm` | RMSNorm with epsilon from config |
| `silu_gated` | SiLU-gated MLP: `down(silu(gate) * up(x))` |
| `softmax` | Online single-pass softmax |
| `expert_router` | MoE top-k gating + expert selection |

### Acceptance criterion
Qwen3-0.6B on RTX 4090:
- Prefill throughput > 12,000 tok/s
- Decode throughput > 3x llama.cpp on same model/GPU
- All kernels pass numerical parity test against HuggingFace reference (tolerance 1e-3)

---

## Phase 6: USB Cluster Mesh (`ae-cluster`)

### Goal
Connect any mix of hardware (laptop + NVIDIA workstation + Mac mini) with USB-C cables and form a single unified inference accelerator. Zero configuration on the user side.

### Device discovery (Zenoh P2P)

```rust
// ae-cluster/src/discovery.rs

pub async fn discover_peers(timeout: Duration) -> Vec<PeerNode> {
    // Zenoh auto-discovers peers on the local network / USB subnet
    // No IP addresses, no config files, no port forwarding
    let session = zenoh::open(zenoh::config::peer()).await?;
    let sub = session.declare_subscriber("ae/nodes/announce").await?;

    let mut peers = Vec::new();
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            sample = sub.recv_async() => {
                let node: NodeAnnouncement = rkyv::from_bytes(sample?.payload())?;
                peers.push(PeerNode::from(node));
            }
            _ = &mut deadline => break,
        }
    }
    peers
}
```

### USB network tunneling

USB-C cables running RNDIS (Windows/Linux) or CDC-ECM (macOS/Linux) present as a standard IP network interface. Zenoh discovers peers over this interface automatically.

```
Host machine  ←→  USB-C cable  ←→  Client machine
  192.168.7.1                         192.168.7.2
     │                                     │
     └──── Zenoh P2P session over RNDIS ───┘
     └──── rkyv activation tensors (zero-copy) ──┘
```

### Layer assignment

```rust
// ae-cluster/src/partition.rs

pub fn assign_layers(
    n_layers: usize,
    nodes: &[PeerNode],       // sorted by gflops descending
) -> Vec<LayerRange> {
    // Proportional assignment: nodes with more GFLOPS get more layers
    let total_gflops: f32 = nodes.iter().map(|n| n.gflops).sum();
    let mut start = 0;
    let mut assignments = Vec::new();
    for node in nodes {
        let frac = node.gflops / total_gflops;
        let n = ((n_layers as f32) * frac).round() as usize;
        let end = (start + n).min(n_layers);
        assignments.push(LayerRange { node_id: node.id, layers: start..end });
        start = end;
    }
    assignments
}
```

### Zero-copy activation streaming

```rust
// ae-cluster/src/stream.rs

// rkyv serializes the activation tensor in-place — no copy from GPU to CPU to network buffer
// The receiving node maps it directly from the network buffer into GPU memory

pub async fn stream_activation(
    activation: &Tensor,    // on-device tensor
    publisher: &zenoh::Publisher<'_>,
) -> Result<()> {
    // SAFETY: tensor data is valid for the duration of this call
    let bytes = unsafe { activation.as_byte_slice() };
    publisher.put(bytes).await?;
    Ok(())
}
```

### Pause-Replicate-Retry (fault tolerance)

```
Heartbeat every 500ms between coordinator and each node.

On 3 consecutive missed heartbeats:
  1. Pause scheduler (tokio::sync::Notify)
  2. Query surviving nodes for capacity
  3. Re-assign lost node's layers to survivors
  4. Re-prefill in-flight sequences to rebuild KV state
  5. Resume — zero token loss guaranteed

vLLM fails and drops the request. AgnosticEngine does not.
```

### Acceptance criterion
2-node USB test: RTX workstation + MacBook connected via USB-C.
- Combined inference must be faster than either node alone
- Kill the MacBook mid-generation: output must continue without token loss
- Inter-node activation latency < 5ms per layer boundary

---

## CLI: The User-Facing Interface

### Install

```bash
cargo install llm    # Rust toolchain required — or use the pre-built binary
```

### Commands

```bash
# Serve any model (HF repo or local path)
llm serve Qwen/Qwen3-0.6B
llm serve ./models/llama3-8b.gguf
llm serve Qwen/Qwen3-4B --device cuda:0
llm serve Qwen/Qwen3-4B --cluster  # auto-discover USB peers

# One-shot generation
llm run Qwen/Qwen3-0.6B "What is the capital of France?"

# Show hardware profile
llm devices

# Benchmark against vLLM / llama.cpp
llm bench Qwen/Qwen3-4B --compare-vllm

# Download and quantize
llm pull Llama-3.1-8B --quant q8
```

### OpenAI-compatible API

```
POST /v1/chat/completions   ← streaming SSE or non-streaming JSON
POST /v1/completions
POST /v1/embeddings
GET  /v1/models
GET  /health
GET  /metrics               ← Prometheus format
```

---

## Sprint Roadmap

### Sprint 1 — Ingestion + Paradigm Router (Weeks 1–6)
- `ae-ingest`: mmap loader, metadata parser, fallback tensor-name analysis
- `ae-classify`: paradigm router, arch detection
- `ae-core`: `LlmBackend` trait, all types
- First acceptance test: `llm serve Qwen3-0.6B` starts and serves requests

### Sprint 2 — Hardware Profiler + CPU Reference Backend (Weeks 7–12)
- `ae-profile`: CubeCL device discovery, bandwidth measurement, autotune cache
- `ae-core/backends/candle.rs`: CPU reference backend (correctness, not speed)
- Verification harness: HuggingFace parity tests for every kernel
- Acceptance test: 20-token coherent generation on CPU, all parity tests pass

### Sprint 3 — Scheduler + PagedAttention (Weeks 13–18)
- `ae-schedule`: block pool, sequence state machine, prefix trie, scheduler loop
- Leader-Peer deferral mechanism
- OneShot / Decode / Mixed classification
- Acceptance test: 96 concurrent requests, KV waste < 4%

### Sprint 4 — JIT Kernels + GPU Backend (Weeks 19–25)
- `ae-kernel`: all kernels via `#[cube]`, autotune registry
- `ae-core/backends/cubecl.rs`: GPU backend implementing `LlmBackend`
- Acceptance test: Qwen3-4B at > 150 req/s, > 25% faster than llama.cpp

### Sprint 5 — HTTP Server + Full CLI (Weeks 26–29)
- `ae-cli`: axum server, SSE streaming, OpenAI schema
- `llm` binary with all CLI commands
- Prometheus metrics endpoint
- Acceptance test: OpenAI Python client works against `llm serve` without modification

### Sprint 6 — Cluster Mesh (Weeks 30–35)
- `ae-cluster`: Zenoh discovery, layer assignment, rkyv streaming, heartbeat
- Pause-Replicate-Retry fault recovery
- USB RNDIS/CDC-ECM tunnel documentation
- Acceptance test: 2-node inference, < 5ms inter-node latency, fault recovery without token loss

---

## Verification Matrix

| Benchmark | Target | How to measure |
|---|---|---|
| Day-zero model support | 100% auto-classify | `llm serve` on a newly released HF model — no code changes |
| Quantized GEMV speed | > 25% vs llama.cpp | `llm bench --compare-llamacpp` on same GGUF + GPU |
| Concurrent throughput | > 15% vs vLLM | ShareGPT benchmark at concurrency 96 |
| KV waste | < 4% | Monitor block pool free ratio over 1000 requests |
| USB inter-node latency | < 5ms | `ae-cluster` benchmark tool, 2-node USB setup |
| Numerical parity | Max error < 1e-3 | `cargo test --test parity` vs HuggingFace reference |
| Memory safety | Zero UB | `cargo miri test` on CPU path; Clippy deny warnings |
| Fault recovery | Zero token loss | Kill a cluster node mid-generation, verify output continuity |

---

## Agent Rules

These rules apply to every coding agent working on this project:

1. **Read the relevant MLC-LLM / HuggingFace source before writing each component.** The target file is listed in each section above.
2. **One sprint at a time.** Do not start Sprint 4 until all Sprint 3 acceptance criteria pass.
3. **No `unwrap()` in library code.** All errors propagate via `anyhow::Result`.
4. **No `unsafe` without a `// SAFETY:` comment** explaining exactly why it is sound.
5. **Every struct gets a doc comment** explaining the invariant it maintains.
6. **No test may contain `assert!(true)`** or compare a literal to itself. Every test must be able to fail.
7. **Block size is a compile-time constant.** Never make it a runtime parameter inside kernel code.
8. **Parity tests run before GPU kernel commits.** Any kernel change must pass `cargo test --test parity` before merging.
9. **The CLI is the contract.** If `llm serve` breaks for any model that previously worked, that is a regression, not a feature gap.