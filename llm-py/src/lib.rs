//! Python bindings for llm-rs, exposing a small, vLLM-style API:
//!
//! ```python
//! from llm_rs import LLM, SamplingParams
//! llm = LLM(model="path/to/model.gguf")
//! outputs = llm.generate(["Hello, how are you?"], SamplingParams(temperature=0.7, max_tokens=64))
//! print(outputs[0].text)
//! ```
//!
//! Binds directly to llm-core/llm-scheduler (not through the C `llm-ffi` layer) so this
//! crate owns its own Tokio runtime and a real `LlmTokenizer`, sidestepping the two
//! FFI-layer bugs (`tokio::spawn` with no reactor present, placeholder char-cast
//! "tokenization") that a prior audit found in `llm-ffi`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_core::tokenizer::LlmTokenizer;
use llm_core::types::{InferRequest, SampleParams, SeqId, TokenId};
use llm_scheduler::engine::{ServingEngine, TokenEvent};

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() > n {
        let t: String = s.chars().take(n).collect();
        format!("{t}...")
    } else {
        s.to_string()
    }
}

/// Sampling hyperparameters for a `generate()` call. Mirrors llm-core's
/// `SampleParams`, plus `max_tokens` (the generation length cap).
#[pyclass(module = "llm_rs")]
#[derive(Clone, Debug)]
pub struct SamplingParams {
    #[pyo3(get, set)]
    pub temperature: f32,
    #[pyo3(get, set)]
    pub top_p: f32,
    #[pyo3(get, set)]
    pub top_k: usize,
    #[pyo3(get, set)]
    pub repetition_penalty: f32,
    #[pyo3(get, set)]
    pub max_tokens: usize,
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature=0.7, top_p=0.9, top_k=0, repetition_penalty=1.1, max_tokens=256))]
    fn new(
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        max_tokens: usize,
    ) -> PyResult<Self> {
        if !(temperature.is_finite()) || temperature < 0.0 {
            return Err(PyValueError::new_err(
                "temperature must be a finite number >= 0.0",
            ));
        }
        if !(0.0..=1.0).contains(&top_p) {
            return Err(PyValueError::new_err("top_p must be between 0.0 and 1.0"));
        }
        if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
            return Err(PyValueError::new_err(
                "repetition_penalty must be a finite number > 0.0",
            ));
        }
        if max_tokens == 0 {
            return Err(PyValueError::new_err("max_tokens must be >= 1"));
        }
        Ok(Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            max_tokens,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SamplingParams(temperature={}, top_p={}, top_k={}, repetition_penalty={}, max_tokens={})",
            self.temperature, self.top_p, self.top_k, self.repetition_penalty, self.max_tokens
        )
    }
}

/// The result of generating from one prompt.
#[pyclass(module = "llm_rs")]
#[derive(Clone)]
pub struct RequestOutput {
    #[pyo3(get)]
    pub prompt: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub token_ids: Vec<u32>,
    #[pyo3(get)]
    pub finish_reason: String,
    #[pyo3(get)]
    pub prefill_tokens_per_sec: f64,
    #[pyo3(get)]
    pub decode_tokens_per_sec: f64,
}

#[pymethods]
impl RequestOutput {
    fn __repr__(&self) -> String {
        format!(
            "RequestOutput(prompt={:?}, text={:?}, finish_reason={:?})",
            truncate(&self.prompt, 40),
            truncate(&self.text, 60),
            self.finish_reason
        )
    }
}

/// A loaded model, ready to serve `generate()` calls. Owns its own Tokio
/// runtime and background scheduler — there is no need for a caller to be
/// inside any async context; every method here is a plain blocking call.
#[pyclass(module = "llm_rs")]
pub struct LLM {
    engine: Arc<ServingEngine>,
    tokenizer: Arc<LlmTokenizer>,
    runtime: tokio::runtime::Runtime,
    next_seq_id: AtomicU64,
    eos_token_id: u32,
    backend_name: String,
    model_name: String,
    vocab_size: usize,
    arch: String,
    has_vision_encoder: bool,
    has_audio_encoder: bool,
}

fn resolve_tokenizer_path(model_path: &Path, tokenizer_path: Option<String>) -> PyResult<PathBuf> {
    if let Some(p) = tokenizer_path {
        let p = PathBuf::from(p);
        if !p.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "tokenizer_path {:?} does not exist",
                p
            )));
        }
        return Ok(p);
    }
    let dir = if model_path.is_file() {
        model_path.parent().ok_or_else(|| {
            PyValueError::new_err(format!(
                "model path {:?} has no parent directory to look for tokenizer.json in \
                 — pass tokenizer_path=... explicitly",
                model_path
            ))
        })?
    } else {
        model_path
    };
    let candidate = dir.join("tokenizer.json");
    if !candidate.exists() {
        return Err(PyFileNotFoundError::new_err(format!(
            "no tokenizer.json found next to {:?} — pass tokenizer_path=... explicitly",
            model_path
        )));
    }
    Ok(candidate)
}

#[pymethods]
impl LLM {
    /// Load a model (GGUF file or HF-style safetensors directory) and start
    /// its serving engine. Hardware backend (CUDA/Metal/CPU) is chosen
    /// automatically by `HardwareProfile`, the same runtime dispatch used by
    /// `llm-cli` — never pinned from Python.
    #[new]
    #[pyo3(signature = (model, tokenizer_path=None, explicit_dequantize=false, use_vram_embeddings=false, block_pool_size=1024))]
    fn new(
        model: String,
        tokenizer_path: Option<String>,
        explicit_dequantize: bool,
        use_vram_embeddings: bool,
        block_pool_size: usize,
    ) -> PyResult<Self> {
        let model_path = Path::new(&model);
        if !model_path.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "model path not found: {model} (pass a GGUF file or a HF-style safetensors \
                 directory containing config.json)"
            )));
        }

        let resolved_tokenizer_path = resolve_tokenizer_path(model_path, tokenizer_path)?;

        let runtime = tokio::runtime::Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("failed to create async runtime: {e}"))
        })?;

        let mut backend = Box::new(CandleBackend::new());
        if explicit_dequantize {
            backend.set_explicit_dequantize(true);
        }
        if use_vram_embeddings {
            backend.set_use_vram_embeddings(true);
        }

        let meta = backend.load_weights(model_path).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to load model weights from {model}: {e:#}"
            ))
        })?;

        let eos_token_id = backend.eos_token_id();
        let backend_name = backend.name().to_string();

        let tokenizer = LlmTokenizer::from_file(&resolved_tokenizer_path).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "failed to load tokenizer from {:?}: {e:#}",
                resolved_tokenizer_path
            ))
        })?;

        // ServingEngine::new spawns its background scheduler loop via
        // tokio::spawn, which requires an active reactor — enter this LLM's
        // own runtime for the duration of construction so that spawn succeeds
        // (this is the fix for the "tokio::spawn with no runtime" class of
        // bug found in the sibling llm-ffi C API).
        let engine = {
            let _guard = runtime.enter();
            ServingEngine::new(backend, block_pool_size)
        };

        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("model")
            .to_string();

        Ok(Self {
            engine: Arc::new(engine),
            tokenizer: Arc::new(tokenizer),
            runtime,
            next_seq_id: AtomicU64::new(1),
            eos_token_id,
            backend_name,
            model_name,
            vocab_size: meta.vocab_size,
            arch: meta.arch.clone(),
            has_vision_encoder: meta.has_vision_encoder,
            has_audio_encoder: meta.has_audio_encoder,
        })
    }

    /// Generate completions for a batch of prompts. Blocking (releases the
    /// GIL while it runs), returns once every prompt has finished.
    #[pyo3(signature = (prompts, sampling_params=None))]
    fn generate(
        &self,
        py: Python<'_>,
        prompts: Vec<String>,
        sampling_params: Option<SamplingParams>,
    ) -> PyResult<Vec<RequestOutput>> {
        if prompts.is_empty() {
            return Ok(Vec::new());
        }
        let params = sampling_params.unwrap_or(SamplingParams {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.1,
            max_tokens: 256,
        });

        let engine = self.engine.clone();
        let tokenizer = self.tokenizer.clone();
        let eos_token_id = self.eos_token_id;

        // Assign seq_ids and pre-tokenize before entering the async block so
        // tokenizer errors surface as a clean PyErr rather than inside block_on.
        let mut order: Vec<SeqId> = Vec::with_capacity(prompts.len());
        let mut requests: Vec<InferRequest> = Vec::with_capacity(prompts.len());
        let mut prompt_lens: HashMap<SeqId, usize> = HashMap::new();
        for prompt in &prompts {
            let prompt_tokens = tokenizer.encode(prompt, true).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to tokenize prompt: {e:#}"))
            })?;
            let seq_id = self.next_seq_id.fetch_add(1, Ordering::Relaxed);
            order.push(seq_id);
            prompt_lens.insert(seq_id, prompt_tokens.len());
            requests.push(InferRequest {
                seq_id,
                prompt_tokens,
                max_new_tokens: params.max_tokens,
                sample_params: SampleParams {
                    temperature: params.temperature,
                    top_p: params.top_p,
                    top_k: params.top_k,
                    repetition_penalty: params.repetition_penalty,
                    max_new_tokens: params.max_tokens,
                },
            });
        }

        // Release the GIL for the duration of generation so other Python
        // threads aren't blocked on this potentially-long call.
        let raw: Vec<(SeqId, Vec<TokenId>, String, f64, f64)> = py
            .allow_threads(|| {
                self.runtime.block_on(async move {
                    run_batch(&engine, eos_token_id, requests, prompt_lens).await
                })
            })
            .map_err(|e| PyRuntimeError::new_err(format!("{e:#}")))?;

        let mut by_seq: HashMap<SeqId, (Vec<TokenId>, String, f64, f64)> = raw
            .into_iter()
            .map(|(seq_id, toks, reason, p_tps, d_tps)| (seq_id, (toks, reason, p_tps, d_tps)))
            .collect();

        let mut outputs = Vec::with_capacity(prompts.len());
        for (prompt, seq_id) in prompts.into_iter().zip(order.into_iter()) {
            let (token_ids, finish_reason, prefill_tps, decode_tps) =
                by_seq.remove(&seq_id).ok_or_else(|| {
                    PyRuntimeError::new_err("internal error: missing output for a submitted request")
                })?;
            let text = tokenizer.decode(&token_ids, true).map_err(|e| {
                PyRuntimeError::new_err(format!("failed to decode output tokens: {e:#}"))
            })?;
            outputs.push(RequestOutput {
                prompt,
                text,
                token_ids,
                finish_reason,
                prefill_tokens_per_sec: prefill_tps,
                decode_tokens_per_sec: decode_tps,
            });
        }
        Ok(outputs)
    }

    #[getter]
    fn model_name(&self) -> &str {
        &self.model_name
    }

    #[getter]
    fn backend_name(&self) -> &str {
        &self.backend_name
    }

    #[getter]
    fn arch(&self) -> &str {
        &self.arch
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    #[getter]
    fn has_vision_encoder(&self) -> bool {
        self.has_vision_encoder
    }

    #[getter]
    fn has_audio_encoder(&self) -> bool {
        self.has_audio_encoder
    }

    fn __repr__(&self) -> String {
        format!(
            "LLM(model_name={:?}, arch={:?}, backend={:?})",
            self.model_name, self.arch, self.backend_name
        )
    }
}

/// Run a batch of requests to completion, returning
/// `(seq_id, token_ids, finish_reason, prefill_tok_s, decode_tok_s)` per request.
///
/// A request finishes with `finish_reason = "stop"` on a normal EOS/max-tokens
/// event. If the broadcast channel reports `Lagged` (this consumer fell behind
/// under heavy concurrent load), affected in-flight requests are reported with
/// `finish_reason = "error: response stream lagged, output may be truncated"`
/// rather than being silently treated as a clean, complete stop — matching the
/// "no silent fallback" fix applied to llm-cli's HTTP server for the same class
/// of bug.
async fn run_batch(
    engine: &ServingEngine,
    eos_token_id: u32,
    requests: Vec<InferRequest>,
    prompt_lens: HashMap<SeqId, usize>,
) -> anyhow::Result<Vec<(SeqId, Vec<TokenId>, String, f64, f64)>> {
    let mut rx = engine.subscribe();
    let mut pending: std::collections::HashSet<SeqId> =
        requests.iter().map(|r| r.seq_id).collect();
    let mut tokens: HashMap<SeqId, Vec<TokenId>> = HashMap::new();
    let mut finish_reason: HashMap<SeqId, String> = HashMap::new();
    let mut first_token_at: HashMap<SeqId, Instant> = HashMap::new();
    let mut last_token_at: HashMap<SeqId, Instant> = HashMap::new();
    let start = Instant::now();

    for req in requests {
        engine.add_request(req)?;
    }

    while !pending.is_empty() {
        match rx.recv().await {
            Ok(TokenEvent {
                seq_id,
                token_id,
                is_eos,
            }) => {
                if !pending.contains(&seq_id) {
                    continue;
                }
                let now = Instant::now();
                first_token_at.entry(seq_id).or_insert(now);
                last_token_at.insert(seq_id, now);
                tokens.entry(seq_id).or_default().push(token_id);
                if is_eos || token_id == eos_token_id {
                    finish_reason.insert(seq_id, "stop".to_string());
                    pending.remove(&seq_id);
                }
            }
            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                // We fell behind the broadcast channel; any request still
                // pending may have missed tokens. Report this explicitly
                // instead of silently returning a (possibly truncated)
                // "stop" — see doc comment above.
                for seq_id in pending.drain() {
                    finish_reason.insert(
                        seq_id,
                        "error: response stream lagged, output may be truncated".to_string(),
                    );
                }
                break;
            }
            Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                for seq_id in pending.drain() {
                    finish_reason
                        .entry(seq_id)
                        .or_insert_with(|| "error: engine shut down mid-generation".to_string());
                }
                break;
            }
        }
    }

    let mut out = Vec::new();
    for (seq_id, toks) in tokens {
        let n_gen = toks.len();
        let prompt_len = prompt_lens.get(&seq_id).copied().unwrap_or(0);
        let ttft = first_token_at
            .get(&seq_id)
            .map(|t| t.duration_since(start).as_secs_f64())
            .unwrap_or(0.0);
        let total = last_token_at
            .get(&seq_id)
            .map(|t| t.duration_since(start).as_secs_f64())
            .unwrap_or(ttft);
        let prefill_tps = if ttft > 0.0 {
            prompt_len as f64 / ttft
        } else {
            0.0
        };
        let decode_dur = (total - ttft).max(1e-9);
        let decode_tps = if n_gen > 1 {
            (n_gen - 1) as f64 / decode_dur
        } else {
            0.0
        };
        let reason = finish_reason
            .remove(&seq_id)
            .unwrap_or_else(|| "stop".to_string());
        out.push((seq_id, toks, reason, prefill_tps, decode_tps));
    }
    Ok(out)
}

#[pymodule]
fn _llm_rs_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    tracing_subscriber::fmt::try_init().ok();
    m.add_class::<LLM>()?;
    m.add_class::<SamplingParams>()?;
    m.add_class::<RequestOutput>()?;
    Ok(())
}
