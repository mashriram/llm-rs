use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use axum::{
    routing::post,
    response::sse::{Event, Sse},
    Json, Router, Extension,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::broadcast;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::limit::RequestBodyLimitLayer;
use tracing::{error, warn, info};

use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_core::tokenizer::LlmTokenizer;
use llm_core::types::{InferRequest, ModelMeta, SampleParams};
use llm_scheduler::engine::{ServingEngine, TokenEvent};

/// Default ceiling on a client-requested `max_tokens` when the operator
/// doesn't override it with `--max-tokens-limit`. Chosen to bound how long a
/// single request can tie up engine resources.
pub const DEFAULT_MAX_TOKENS_LIMIT: usize = 4096;

/// Default cap on request body size (bytes) to stop a client from sending an
/// enormous body to exhaust server memory. `tower_http::limit::RequestBodyLimitLayer`
/// enforces this before the body is buffered/parsed.
pub const DEFAULT_MAX_BODY_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Serialize)]
pub struct DeltaMessage {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: DeltaMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
}

pub struct AppState {
    pub engine: Arc<ServingEngine>,
    pub model_name: String,
    pub tokenizer: Arc<LlmTokenizer>,
    /// The served model's own metadata (architecture, chat template, etc.),
    /// so requests can be formatted with the MODEL'S chat template instead
    /// of a one-size-fits-all hardcoded format.
    pub meta: Arc<ModelMeta>,
    /// Upper bound on a client-requested `max_tokens`; requests above this
    /// are rejected with 400 Bad Request rather than silently clamped.
    pub max_tokens_limit: usize,
}

use axum::routing::get;

#[derive(Debug, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

/// Build the router with the default request body size cap
/// ([`DEFAULT_MAX_BODY_BYTES`]). Use [`create_router_with_body_limit`] to
/// override it (e.g. from a `--max-body-bytes` CLI flag).
pub fn create_router(state: Arc<AppState>) -> Router {
    create_router_with_body_limit(state, DEFAULT_MAX_BODY_BYTES)
}

pub fn create_router_with_body_limit(state: Arc<AppState>, max_body_bytes: usize) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(Extension(state))
        // Caps request body size so a client can't exhaust server memory
        // with an oversized request. Must sit "outside" body extraction.
        .layer(RequestBodyLimitLayer::new(max_body_bytes))
}

pub async fn health() -> &'static str {
    "OK"
}

pub async fn models(
    Extension(state): Extension<Arc<AppState>>,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model".to_string(),
            created: 1686935002,
            owned_by: "llm-rs".to_string(),
        }],
    })
}

fn bad_request_json(message: &str, error_type: &str) -> axum::response::Response {
    axum::response::IntoResponse::into_response((
        axum::http::StatusCode::BAD_REQUEST,
        Json(json!({ "error": { "message": message, "type": error_type } })),
    ))
}

fn service_unavailable_json(message: &str) -> axum::response::Response {
    axum::response::IntoResponse::into_response((
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({ "error": { "message": message, "type": "stream_lagged" } })),
    ))
}

pub async fn chat_completions(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    // Reject (not silently clamp) requests asking for more tokens than the
    // configured ceiling: a client requesting `max_tokens: 999999999` would
    // otherwise tie up engine resources indefinitely.
    if let Some(requested) = req.max_tokens {
        if requested > state.max_tokens_limit {
            return bad_request_json(
                &format!(
                    "max_tokens ({}) exceeds this server's limit of {}",
                    requested, state.max_tokens_limit
                ),
                "invalid_request_error",
            );
        }
    }

    let seq_id = rand::random::<u64>();

    let prompt = render_prompt(&req.messages, &state.meta);
    let prompt_tokens = match state.tokenizer.encode(&prompt, true) {
        Ok(mut t) => {
            maybe_prepend_bos(&mut t, &state.meta);
            t
        }
        Err(e) => {
            error!("Failed to encode prompt: {:?}", e);
            return axum::response::IntoResponse::into_response(
                (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Failed to encode prompt")
            );
        }
    };

    let infer_req = InferRequest {
        seq_id,
        prompt_tokens,
        sample_params: SampleParams {
            temperature: req.temperature.unwrap_or(0.7),
            top_p: req.top_p.unwrap_or(0.9),
            top_k: 40,
            repetition_penalty: 1.1,
            max_new_tokens: req.max_tokens.unwrap_or(128),
        },
        max_new_tokens: req.max_tokens.unwrap_or(128),
    };

    let mut token_rx = state.engine.subscribe();

    if let Err(e) = state.engine.add_request(infer_req) {
        error!("Failed to add request: {:?}", e);
        return axum::response::IntoResponse::into_response(
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
        );
    }

    // A clock set before the UNIX epoch would make `duration_since` fail; fall back
    // to 0 rather than panicking the request handler over a misconfigured clock.
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs();
    let model_name = state.model_name.clone();
    let tokenizer = state.tokenizer.clone();

    if req.stream {
        // `token_rx` is a broadcast receiver whose sender lives for the whole
        // server process, so a combinator chain built directly on top of it
        // (e.g. `take_while`) can only stop *after* observing one more item —
        // which may never arrive once this seq_id's generation is done,
        // hanging the HTTP response forever. Instead, forward through a
        // dedicated mpsc channel from a background task that explicitly
        // breaks (dropping the sender, which ends the stream) the moment it
        // sees `is_eos`, with no dependency on further broadcast traffic.
        let (tx, rx) = tokio::sync::mpsc::channel::<Event>(16);
        tokio::spawn(async move {
            loop {
                match token_rx.recv().await {
                    Ok(event) if event.seq_id == seq_id => {
                        let text = tokenizer.decode(&[event.token_id], true).unwrap_or_else(|_| " ".to_string());
                        let is_eos = event.is_eos;
                        let chunk = ChatCompletionChunk {
                            id: format!("chatcmpl-{}", seq_id),
                            object: "chat.completion.chunk".to_string(),
                            created,
                            model: model_name.clone(),
                            choices: vec![ChatChunkChoice {
                                index: 0,
                                delta: DeltaMessage {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: if is_eos { Some("stop".to_string()) } else { None },
                            }],
                        };
                        match serde_json::to_string(&chunk) {
                            Ok(json) => {
                                if tx.send(Event::default().data(json)).await.is_err() {
                                    break; // client disconnected
                                }
                            }
                            Err(e) => {
                                error!("Failed to serialize chat completion chunk: {:?}", e);
                            }
                        }
                        if is_eos {
                            // OpenAI-convention end-of-stream sentinel so clients
                            // (curl/openai-python) see an explicit close instead
                            // of hanging after the last content chunk.
                            let _ = tx.send(Event::default().data("[DONE]")).await;
                            break;
                        }
                    }
                    Ok(_) => continue, // a different request's token, keep waiting
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!("stream for seq_id {} lagged behind by {} events; truncating with an error event", seq_id, n);
                        let _ = tx.send(Event::default().event("error").data(
                            json!({ "error": { "message": "response stream fell behind and was truncated, please retry", "type": "stream_lagged" } }).to_string()
                        )).await;
                        break;
                    }
                    Err(broadcast::error::RecvError::Closed) => break, // clean shutdown
                }
            }
        });

        let stream = ReceiverStream::new(rx).map(|ev| Ok::<Event, std::convert::Infallible>(ev));
        axum::response::IntoResponse::into_response(Sse::new(stream))
    } else {
        let mut full_tokens = Vec::new();
        loop {
            match token_rx.recv().await {
                Ok(event) if event.seq_id == seq_id => {
                    full_tokens.push(event.token_id);
                    if event.is_eos {
                        break;
                    }
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    // Distinguish a lagged/truncated response from a clean
                    // close: silently returning a truncated 200 here would
                    // report success for output that is missing tokens.
                    warn!("non-streaming response for seq_id {} lagged behind by {} events; output truncated", seq_id, n);
                    return service_unavailable_json(
                        "response stream fell behind and was truncated, please retry",
                    );
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }

        let full_text = tokenizer.decode(&full_tokens, true).unwrap_or_else(|_| "".to_string());

        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", seq_id),
            object: "chat.completion".to_string(),
            created,
            model: state.model_name.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: full_text,
                },
                finish_reason: "stop".to_string(),
            }],
        };

        axum::response::IntoResponse::into_response(Json(response))
    }
}

// ---------------------------------------------------------------------------
// Shared chat-template rendering (used by both the HTTP server above and the
// `chat` TUI binary) — see CLAUDE.md's "one canonical X" rule and audit
// finding #7/#9/#10: this used to be duplicated (and out of sync) between
// this file (hardcoded ChatML) and bin/chat.rs (real per-arch rendering).
// ---------------------------------------------------------------------------

/// Detect the chat format from `ModelMeta` and try to render via Jinja, with
/// a robust per-architecture fallback if there's no template or it fails to
/// render.
pub fn render_prompt(messages: &[ChatMessage], meta: &ModelMeta) -> String {
    if let Some(template_str) = &meta.chat_template {
        match try_jinja_render(template_str, messages) {
            Ok(rendered) => return rendered,
            Err(e) => {
                warn!("Jinja chat-template render failed ({}), falling back to manual format", e);
            }
        }
    }
    manual_format(messages, meta)
}

fn try_jinja_render(template_str: &str, messages: &[ChatMessage]) -> anyhow::Result<String> {
    let mut env = minijinja::Environment::new();
    // Add Python-compat builtins (tojson, etc.)
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // Register strftime_now: returns today's date string (HF templates use this)
    env.add_function("strftime_now", |fmt: &str| -> String {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        let days = secs / 86400;
        let year = 1970 + (days / 365) as u32;
        let day_of_year = days % 365;
        let month = (day_of_year / 30 + 1).min(12) as u32;
        let day = (day_of_year % 30 + 1) as u32;
        if fmt.contains("%Y") || fmt.contains("%d") {
            format!("{:02} {:?} {}", day, month, year)
        } else {
            format!("{}-{:02}-{:02}", year, month, day)
        }
    });

    // Strip HuggingFace-specific non-standard Jinja2 tags that minijinja doesn't support
    let cleaned = template_str
        .replace("{% generation %}", "")
        .replace("{% endgeneration %}", "");

    let msgs: Vec<minijinja::Value> = messages.iter().map(|m| {
        minijinja::context! {
            role => m.role.as_str(),
            content => m.content.as_str(),
        }
    }).collect();

    env.add_template("chat", &cleaned)?;
    let tmpl = env.get_template("chat")?;

    let ctx = minijinja::context! {
        messages => msgs,
        add_generation_prompt => true,
        enable_thinking => false,
    };

    let rendered = tmpl.render(ctx)?;
    Ok(rendered)
}

pub fn manual_format(messages: &[ChatMessage], meta: &ModelMeta) -> String {
    let mut result = String::new();

    if meta.arch == "gemma4" {
        // Gemma 4 format: <|turn>role\ncontent<turn|>\n
        for msg in messages {
            let role = if msg.role == "assistant" { "model" } else { &msg.role };
            result.push_str(&format!("<|turn>{}\n{}<turn|>\n", role, msg.content));
        }
        result.push_str("<|turn>model\n");
    } else if meta.is_gemma {
        // Gemma 1/2 format: <start_of_turn>role\ncontent<end_of_turn>\n
        for msg in messages {
            let role = if msg.role == "assistant" { "model" } else { &msg.role };
            result.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n", role, msg.content));
        }
        result.push_str("<start_of_turn>model\n");
    } else if meta.chat_template.as_deref().map(|t| t.contains("<|im_start|>")).unwrap_or(false)
        || matches!(meta.arch.as_str(), "qwen2" | "qwen3" | "qwen3vl" | "qwen2vl" | "smollm3" | "llama" | "mistral") {
        // ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        for msg in messages {
            result.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
        }
        result.push_str("<|im_start|>assistant\n");
    } else {
        // Generic: role: content\n\n
        for msg in messages {
            if msg.role == "system" {
                result.push_str(&format!("System: {}\n\n", msg.content));
            } else if msg.role == "user" {
                result.push_str(&format!("User: {}\n", msg.content));
            } else {
                result.push_str(&format!("Assistant: {}\n", msg.content));
            }
        }
        result.push_str("Assistant: ");
    }

    result
}

/// Insert BOS token for Gemma if needed. The single canonical copy of this
/// logic (previously duplicated independently in `chat.rs` and
/// `run_model.rs`, with `benchmark_speed.rs` silently missing it).
pub fn maybe_prepend_bos(tokens: &mut Vec<u32>, meta: &ModelMeta) {
    const GEMMA_BOS_TOKEN_ID: u32 = 2;
    if meta.is_gemma && (tokens.is_empty() || tokens[0] != GEMMA_BOS_TOKEN_ID) {
        tokens.insert(0, GEMMA_BOS_TOKEN_ID);
    }
}

// ---------------------------------------------------------------------------
// Shared model-load / generation-loop helpers (audit finding #9): factors out
// the near-identical backend-load -> EOS-resolution -> generate-loop pattern
// that was independently duplicated across run_model.rs, benchmark_speed.rs,
// and chat.rs.
// ---------------------------------------------------------------------------

/// Load a `CandleBackend` from `model_path`, applying the given flags.
/// Returns an actionable error (not a raw io::Error debug-print) if the path
/// doesn't exist.
pub fn load_candle_backend(
    model_path: &Path,
    explicit_dequantize: bool,
    use_vram_embeddings: bool,
    mmproj_path: Option<PathBuf>,
) -> anyhow::Result<(Box<dyn LlmBackend>, ModelMeta)> {
    if !model_path.exists() {
        anyhow::bail!(
            "model file not found at {}: pass a valid --model-path",
            model_path.display()
        );
    }
    let mut backend = Box::new(CandleBackend::new());
    if explicit_dequantize {
        backend.set_explicit_dequantize(true);
    }
    if use_vram_embeddings {
        backend.set_use_vram_embeddings(true);
    }
    if let Some(p) = mmproj_path {
        backend.set_mmproj_path(p);
    }
    let meta = backend.load_weights(model_path).map_err(|e| {
        anyhow::anyhow!(
            "failed to load model weights from {}: {}",
            model_path.display(),
            e
        )
    })?;
    Ok((backend, meta))
}

/// Resolve the set of token ids that should terminate generation: the
/// backend's own EOS id (if it could determine one — see
/// `LlmBackend::eos_token_id`'s doc comment; a model whose metadata omits
/// this is not assumed to use Llama's `2`), plus any of the common
/// chat-turn end markers that exist in this tokenizer's vocabulary.
pub fn resolve_eos_token_ids(backend: &dyn LlmBackend, tokenizer: &LlmTokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    if let Some(id) = backend.eos_token_id() {
        ids.push(id);
    } else {
        tracing::warn!(
            "no EOS token id could be determined for this model; relying on \
             --max-new-tokens and known chat-template stop strings only"
        );
    }
    for tok in &["<|im_end|>", "<end_of_turn>", "<turn|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            ids.push(id);
        }
    }
    ids
}

/// Resolve the tokenizer.json path for `model_path`: look for `tokenizer.json`
/// next to the model file (or inside the model directory). No model-specific
/// guessing — if it isn't found, return a clear, actionable error telling the
/// caller to pass `--tokenizer-path` explicitly (audit finding #6: this used
/// to hardcode fallback guesses for two specific Qwen checkpoints).
pub fn resolve_tokenizer_path(model_path: &Path) -> anyhow::Result<PathBuf> {
    let candidate = if model_path.is_file() {
        let parent = model_path.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "model_path {:?} has no parent directory to look for a tokenizer in; pass --tokenizer-path explicitly",
                model_path
            )
        })?;
        parent.join("tokenizer.json")
    } else {
        model_path.join("tokenizer.json")
    };

    if candidate.exists() {
        Ok(candidate)
    } else {
        anyhow::bail!(
            "no tokenizer.json found at {} (next to model_path {}); pass --tokenizer-path explicitly with the path to a compatible tokenizer.json",
            candidate.display(),
            model_path.display()
        )
    }
}

/// Result of draining one sequence's generation to completion.
#[derive(Debug, Default)]
pub struct GenLoopResult {
    pub tokens: Vec<u32>,
    pub ttft: Option<Duration>,
    pub total_dur: Duration,
    /// Set if the broadcast receiver fell behind (lagged) — the returned
    /// `tokens` may be an incomplete/truncated view of what was generated.
    pub lagged: bool,
}

/// Drain generation events for `seq_id` off `rx` until EOS, `max_new_tokens`
/// is hit, or the channel closes/lags, invoking `on_token(token_id, decoded_text)`
/// for each token as it arrives (e.g. to print it immediately). This is the
/// one shared implementation of the token-receive loop duplicated (with
/// behavioral drift, e.g. missing BOS handling) across `run_model.rs`,
/// `benchmark_speed.rs`, and `chat.rs`.
pub async fn consume_generation<F: FnMut(u32, &str)>(
    rx: &mut broadcast::Receiver<TokenEvent>,
    seq_id: u64,
    tokenizer: &LlmTokenizer,
    eos_token_ids: &[u32],
    max_new_tokens: usize,
    mut on_token: F,
) -> GenLoopResult {
    let start = Instant::now();
    let mut result = GenLoopResult::default();

    loop {
        match rx.recv().await {
            Ok(event) if event.seq_id == seq_id => {
                if result.ttft.is_none() {
                    result.ttft = Some(start.elapsed());
                }
                let text = tokenizer.decode(&[event.token_id], true).unwrap_or_default();
                on_token(event.token_id, &text);
                result.tokens.push(event.token_id);

                if event.is_eos || eos_token_ids.contains(&event.token_id) {
                    break;
                }
                if result.tokens.len() >= max_new_tokens {
                    break;
                }
            }
            Ok(_) => continue, // a different sequence's token
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("generation loop for seq_id {} lagged by {} events; output may be incomplete", seq_id, n);
                result.lagged = true;
                break;
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }

    result.total_dur = start.elapsed();
    result
}

/// Print the standard TTFT/prefill/decode/overall throughput block shared by
/// `run_model.rs` and `benchmark_speed.rs`.
pub fn print_gen_stats(prompt_len: usize, result: &GenLoopResult) {
    println!("--------------------------------------");
    println!("Benchmark Results (llm-rs):");
    println!("--------------------------------------");
    if let Some(t) = result.ttft {
        println!("Time to First Token (TTFT): {:.2?}", t);
        if t.as_secs_f64() > 0.0 {
            println!("Prefill Speed: {:.2} tokens/sec", prompt_len as f64 / t.as_secs_f64());
        }
        let n_gen = result.tokens.len();
        let decode_dur = result.total_dur.saturating_sub(t);
        if n_gen > 1 && decode_dur.as_secs_f64() > 0.0 {
            println!("Decode Duration: {:.2?}", decode_dur);
            println!("Decode Speed: {:.2} tokens/sec", (n_gen - 1) as f64 / decode_dur.as_secs_f64());
        }
    }
    if result.lagged {
        println!("WARNING: event stream lagged; token/timing counts above may be incomplete.");
    }
    println!("Total Generation Time: {:.2?}", result.total_dur);
    println!("Total Tokens Generated: {}", result.tokens.len());
    if result.total_dur.as_secs_f64() > 0.0 {
        println!("Overall Speed: {:.2} tokens/sec", result.tokens.len() as f64 / result.total_dur.as_secs_f64());
    }
    println!("--------------------------------------");
}

// ---------------------------------------------------------------------------
// Compiled-accelerator-support transparency (audit finding #8): default
// `cargo build` compiles in no CUDA/Metal candle support at all (the `cuda`
// / `metal` Cargo features gate it out), so a HardwareProfile that detects a
// GPU has no code path to actually use it unless the binary was built with
// the matching feature. That's a legitimate build-size/toolchain tradeoff,
// but it must never be a SILENT surprise to whoever runs the binary.
// ---------------------------------------------------------------------------

/// Log, once at startup, which accelerator backends this binary was actually
/// compiled with support for, and warn loudly if the detected hardware
/// doesn't match what's compiled in.
pub fn log_compiled_backend_support() {
    let cuda_compiled = cfg!(feature = "cuda");
    let metal_compiled = cfg!(feature = "metal");

    info!(
        "Compiled accelerator support: cuda={} metal={} (cpu is always available)",
        cuda_compiled, metal_compiled
    );

    let profile = llm_core::profile::HardwareProfile::get();
    match profile.backend {
        llm_core::profile::BackendChoice::Cuda if !cuda_compiled => {
            warn!(
                "HardwareProfile detected a CUDA-capable GPU, but this binary was compiled WITHOUT \
                 --features cuda. Only CPU inference is available regardless of detected hardware. \
                 Rebuild with --features cuda to use the GPU."
            );
        }
        llm_core::profile::BackendChoice::Metal if !metal_compiled => {
            warn!(
                "HardwareProfile detected Metal (Apple Silicon), but this binary was compiled WITHOUT \
                 --features metal. Only CPU inference is available regardless of detected hardware. \
                 Rebuild with --features metal to use the GPU."
            );
        }
        _ => {
            info!("Detected hardware backend ({:?}) matches compiled-in support.", profile.backend);
        }
    }
}
