use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Mutex;
use libc::c_void;

use llm_core::backends::candle::CandleBackend;
use llm_core::backend::LlmBackend;
use llm_core::tokenizer::LlmTokenizer;
use llm_core::types::{InferRequest, SampleParams};
use llm_scheduler::engine::{ServingEngine, TokenEvent};

/// Sentinel returned by [`send_request`] to signal failure. `0` is a valid
/// (if astronomically unlikely) random `seq_id`, so it cannot double as an
/// error code; `u64::MAX` is reserved as the error sentinel and is never
/// handed out as a real `seq_id` (see `next_seq_id`).
pub const INVALID_SEQ_ID: u64 = u64::MAX;

/// Configuration for [`create_engine`].
///
/// # ABI note
/// `tokenizer_path` and `use_dummy` were added after the initial release.
/// Callers linking against the older 2-field layout MUST be recompiled:
/// this is a breaking ABI change, not additive. `tokenizer_path` is
/// mandatory (non-null) unless `use_dummy` is `true`.
#[repr(C)]
pub struct EngineConfig {
    pub model_path: *const c_char,
    /// Path to a `tokenizer.json` compatible with `llm_core::tokenizer::LlmTokenizer`.
    /// May be null only if `use_dummy` is `true`.
    pub tokenizer_path: *const c_char,
    pub block_pool_size: usize,
    /// Explicit, unambiguous opt-in to a non-functional dummy backend for
    /// testing the FFI plumbing without real weights. This is never inferred
    /// from `model_path` (a real model path could legitimately contain the
    /// substrings "dummy"/"tmp"/"temp", e.g. `/tmp/models/llama.gguf`).
    pub use_dummy: bool,
}

#[repr(C)]
pub struct ChatRequest {
    pub prompt: *const c_char,
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

#[repr(C)]
pub struct GenerationResult {
    pub token: u32,
    pub text: *mut c_char,
    pub is_eos: bool,
    /// `true` if this result represents a real error (bad handle, unknown
    /// seq_id, internal panic) rather than a clean end-of-stream. Callers
    /// should check this before treating `is_eos` as "generation finished
    /// normally".
    pub is_error: bool,
}

impl GenerationResult {
    fn error() -> Self {
        GenerationResult { token: 0, text: std::ptr::null_mut(), is_eos: true, is_error: true }
    }

    fn done() -> Self {
        GenerationResult { token: 0, text: std::ptr::null_mut(), is_eos: true, is_error: false }
    }
}

/// Owns the Tokio runtime the whole engine lives on, so `extern "C"`
/// functions (which cannot themselves be `async` and have no caller-provided
/// runtime) always have a live reactor to `spawn`/`block_on` against.
pub struct FfiEngineContext {
    runtime: tokio::runtime::Runtime,
    engine: ServingEngine,
    tokenizer: Option<std::sync::Arc<LlmTokenizer>>,
    /// One dedicated per-request receiver per in-flight `seq_id`, populated
    /// by `send_request` and drained by `poll_token`. Using a dedicated
    /// channel per sequence (mirroring the HTTP server's per-request
    /// `engine.subscribe()` pattern in `llm-cli`) means `poll_token` for one
    /// seq_id can never silently drop another seq_id's tokens.
    pending: Mutex<HashMap<u64, tokio::sync::mpsc::Receiver<TokenEvent>>>,
}

fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller contract on all public FFI functions requires
    // null-terminated C strings for any non-null char pointer.
    unsafe { CStr::from_ptr(ptr) }.to_str().ok()
}

/// Create a new serving engine instance.
///
/// Returns null on any failure: null/invalid `model_path`, null
/// `tokenizer_path` (unless `use_dummy`), tokenizer load failure, or model
/// weight load failure. There is no silent fallback to a dummy backend on
/// real-model load failure — a null return is the only failure signal.
///
/// # SAFETY: The caller must ensure `config.model_path` and
/// `config.tokenizer_path` (when non-null) point to valid null-terminated C
/// strings for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn create_engine(config: EngineConfig) -> *mut c_void {
    // The whole body runs inside catch_unwind: a panic anywhere below
    // (including inside llm-core/llm-scheduler code we call into) must never
    // unwind across this extern "C" boundary, which is documented UB.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        create_engine_impl(config)
    }));
    match result {
        Ok(ptr) => ptr,
        Err(_) => {
            eprintln!("[llm-ffi] panic caught in create_engine; returning null");
            std::ptr::null_mut()
        }
    }
}

fn create_engine_impl(config: EngineConfig) -> *mut c_void {
    let model_path = match cstr_to_str(config.model_path) {
        Some(s) => s,
        None => {
            eprintln!("[llm-ffi] create_engine: model_path is null or not valid UTF-8");
            return std::ptr::null_mut();
        }
    };

    let tokenizer = if config.use_dummy {
        None
    } else {
        let tok_path = match cstr_to_str(config.tokenizer_path) {
            Some(s) => s,
            None => {
                eprintln!("[llm-ffi] create_engine: tokenizer_path is required unless use_dummy is set");
                return std::ptr::null_mut();
            }
        };
        match LlmTokenizer::from_file(tok_path) {
            Ok(t) => Some(std::sync::Arc::new(t)),
            Err(e) => {
                eprintln!("[llm-ffi] create_engine: failed to load tokenizer from {:?}: {:?}", tok_path, e);
                return std::ptr::null_mut();
            }
        }
    };

    let backend: Box<dyn LlmBackend> = if config.use_dummy {
        Box::new(llm_core::backend::DummyBackend::new())
    } else {
        let mut backend = Box::new(CandleBackend::new());
        if let Err(e) = backend.load_weights(std::path::Path::new(model_path)) {
            eprintln!("[llm-ffi] create_engine: failed to load model weights from {:?}: {:?}", model_path, e);
            return std::ptr::null_mut();
        }
        backend
    };

    // Multi-thread runtime: ServingEngine::new spawns a background driver
    // task and send_request spawns a per-request forwarding task, both of
    // which need worker threads distinct from whatever thread the C caller
    // invokes these FFI functions from.
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("[llm-ffi] create_engine: failed to build Tokio runtime: {:?}", e);
            return std::ptr::null_mut();
        }
    };

    // ServingEngine::new calls tokio::spawn synchronously (it is not an
    // async fn) which panics with "there is no reactor running" unless a
    // runtime context is entered on this thread. `enter()` installs that
    // context for the duration of the guard without requiring us to
    // `block_on` anything.
    let engine = {
        let _guard = runtime.enter();
        ServingEngine::new(backend, config.block_pool_size)
    };

    let context = Box::new(FfiEngineContext {
        runtime,
        engine,
        tokenizer,
        pending: Mutex::new(HashMap::new()),
    });
    Box::into_raw(context) as *mut c_void
}

/// Destroy a serving engine instance.
/// # SAFETY: The caller must pass a valid pointer returned by create_engine,
/// not used by any other thread concurrently, and not used again afterwards.
#[no_mangle]
pub unsafe extern "C" fn destroy_engine(engine: *mut c_void) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !engine.is_null() {
            // SAFETY: caller contract above.
            let _ = Box::from_raw(engine as *mut FfiEngineContext);
        }
    }));
}

/// Send a generation request to the engine.
///
/// Returns the assigned `seq_id` on success, or [`INVALID_SEQ_ID`]
/// (`u64::MAX`) on any failure. `0` is a valid, real `seq_id` and must not be
/// treated as an error.
///
/// # SAFETY: The caller must pass a valid engine pointer (from
/// `create_engine`, not yet destroyed) and `request.prompt` must be a valid
/// null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn send_request(engine: *mut c_void, request: ChatRequest) -> u64 {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        send_request_impl(engine, request)
    }));
    result.unwrap_or_else(|_| {
        eprintln!("[llm-ffi] panic caught in send_request; returning INVALID_SEQ_ID");
        INVALID_SEQ_ID
    })
}

fn next_seq_id() -> u64 {
    loop {
        let candidate = rand::random::<u64>();
        if candidate != INVALID_SEQ_ID {
            return candidate;
        }
    }
}

fn send_request_impl(engine: *mut c_void, request: ChatRequest) -> u64 {
    if engine.is_null() {
        return INVALID_SEQ_ID;
    }
    // SAFETY: caller contract on send_request.
    let context = unsafe { &mut *(engine as *mut FfiEngineContext) };

    let prompt = match cstr_to_str(request.prompt) {
        Some(s) => s,
        None => return INVALID_SEQ_ID,
    };

    let tokenizer = match &context.tokenizer {
        Some(t) => t.clone(),
        None => {
            eprintln!("[llm-ffi] send_request: engine has no tokenizer loaded (use_dummy engine?)");
            return INVALID_SEQ_ID;
        }
    };

    let prompt_tokens = match tokenizer.encode(prompt, true) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[llm-ffi] send_request: failed to tokenize prompt: {:?}", e);
            return INVALID_SEQ_ID;
        }
    };

    let seq_id = next_seq_id();

    let infer_req = InferRequest {
        seq_id,
        prompt_tokens,
        sample_params: SampleParams {
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: 40,
            repetition_penalty: 1.1,
            max_new_tokens: request.max_tokens,
        },
        max_new_tokens: request.max_tokens,
    };

    // Subscribe BEFORE add_request so no token can be generated and missed
    // between subscribing and the request landing in the scheduler.
    let mut broadcast_rx = context.engine.subscribe();
    if context.engine.add_request(infer_req).is_err() {
        return INVALID_SEQ_ID;
    }

    // Dedicated per-request channel: a background task filters the shared
    // broadcast stream down to just this seq_id's events. This means
    // poll_token for a different concurrently in-flight seq_id can never
    // steal or drop this sequence's tokens (the old shared-mpsc design did).
    let (tx, rx) = tokio::sync::mpsc::channel::<TokenEvent>(256);
    context.runtime.spawn(async move {
        loop {
            match broadcast_rx.recv().await {
                Ok(event) if event.seq_id == seq_id => {
                    let is_eos = event.is_eos;
                    if tx.send(event).await.is_err() {
                        break; // receiver side (poll_token) dropped
                    }
                    if is_eos {
                        break;
                    }
                }
                Ok(_) => continue, // a different sequence's token
                Err(_) => break,   // broadcast closed or this receiver lagged
            }
        }
    });

    if let Ok(mut map) = context.pending.lock() {
        map.insert(seq_id, rx);
    } else {
        return INVALID_SEQ_ID;
    }

    seq_id
}

/// Poll the next token event for a sequence, blocking the calling thread
/// until one is available (this is a synchronous, C-callable function; a C
/// caller cannot `.await`, so we `block_on` the owned runtime here).
///
/// # SAFETY: The caller must pass a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn poll_token(engine: *mut c_void, seq_id: u64) -> GenerationResult {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        poll_token_impl(engine, seq_id)
    }));
    result.unwrap_or_else(|_| {
        eprintln!("[llm-ffi] panic caught in poll_token; returning error result");
        GenerationResult::error()
    })
}

fn poll_token_impl(engine: *mut c_void, seq_id: u64) -> GenerationResult {
    if engine.is_null() {
        return GenerationResult::error();
    }
    // SAFETY: caller contract on poll_token.
    let context = unsafe { &mut *(engine as *mut FfiEngineContext) };

    let tokenizer = match &context.tokenizer {
        Some(t) => t.clone(),
        None => return GenerationResult::error(),
    };

    // Take the receiver out of the map for the duration of the blocking
    // recv rather than holding the mutex locked while blocked: otherwise a
    // slow/absent producer for seq_id A would stall poll_token calls for an
    // unrelated seq_id B.
    let mut rx = match context.pending.lock() {
        Ok(mut map) => match map.remove(&seq_id) {
            Some(rx) => rx,
            None => return GenerationResult::done(), // unknown or already-finished seq_id
        },
        Err(_) => return GenerationResult::error(),
    };

    let recv_result = context.runtime.block_on(rx.recv());

    match recv_result {
        Some(event) => {
            if !event.is_eos {
                // Still more tokens coming: put the receiver back.
                if let Ok(mut map) = context.pending.lock() {
                    map.insert(seq_id, rx);
                }
            }
            let text = tokenizer.decode(&[event.token_id], true).unwrap_or_default();
            let c_string = match CString::new(text) {
                Ok(c) => c,
                Err(_) => CString::new(" ").unwrap_or_default(),
            };
            GenerationResult {
                token: event.token_id,
                text: c_string.into_raw(),
                is_eos: event.is_eos,
                is_error: false,
            }
        }
        None => GenerationResult::done(),
    }
}

/// Free a string allocated by the FFI.
/// # SAFETY: The caller must pass a valid pointer returned by poll_token, or null.
#[no_mangle]
pub unsafe extern "C" fn free_string(s: *mut c_char) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !s.is_null() {
            // SAFETY: caller contract above; `s` came from `CString::into_raw`.
            let _ = CString::from_raw(s);
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn engine_config(use_dummy: bool, model: &CString, tokenizer: Option<&CString>) -> EngineConfig {
        EngineConfig {
            model_path: model.as_ptr(),
            tokenizer_path: tokenizer.map(|t| t.as_ptr()).unwrap_or(std::ptr::null()),
            block_pool_size: 16,
            use_dummy,
        }
    }

    /// Fix #3: create_engine must never silently infer a dummy backend from
    /// the model path; a legitimate path containing "tmp"/"dummy"/"temp"
    /// (e.g. under /tmp) must still attempt to load a real model.
    #[test]
    fn dummy_backend_requires_explicit_opt_in() {
        let model = CString::new("/tmp/models/definitely-not-real.gguf").unwrap();
        // use_dummy = false and a bogus path: real load must fail closed
        // (null), never silently substitute a working-looking dummy engine.
        let config = engine_config(false, &model, None);
        let ptr = unsafe { create_engine(config) };
        assert!(ptr.is_null(), "a bogus, non-dummy model path must fail to null, not silently use DummyBackend");
    }

    /// Explicit opt-in (`use_dummy: true`) must succeed even with no
    /// tokenizer path and even though the model path doesn't exist.
    #[test]
    fn dummy_backend_explicit_opt_in_succeeds() {
        let model = CString::new("/nonexistent/model.gguf").unwrap();
        let config = engine_config(true, &model, None);
        let ptr = unsafe { create_engine(config) };
        assert!(!ptr.is_null(), "use_dummy:true must succeed regardless of model_path");
        unsafe { destroy_engine(ptr) };
    }

    /// Fix #14: the send_request failure sentinel must never collide with a
    /// real seq_id space in a way that's ambiguous; specifically send_request
    /// on a null engine must return INVALID_SEQ_ID (u64::MAX), not 0.
    #[test]
    fn send_request_on_null_engine_returns_sentinel_not_zero() {
        let prompt = CString::new("hello").unwrap();
        let req = ChatRequest { prompt: prompt.as_ptr(), temperature: 0.7, top_p: 0.9, max_tokens: 8 };
        let seq_id = unsafe { send_request(std::ptr::null_mut(), req) };
        assert_eq!(seq_id, INVALID_SEQ_ID);
        assert_ne!(seq_id, 0, "0 is a valid seq_id and must not double as the error sentinel");
    }

    /// Fix #1: create_engine builds and owns its own Tokio runtime, so
    /// calling it from a plain (non-async, no-runtime) test/thread must not
    /// panic with "there is no reactor running".
    #[test]
    fn create_engine_works_with_no_ambient_tokio_runtime() {
        let model = CString::new("dummy").unwrap();
        let config = engine_config(true, &model, None);
        let ptr = unsafe { create_engine(config) };
        assert!(!ptr.is_null());
        unsafe { destroy_engine(ptr) };
    }

    /// Fix #1: a panic inside an extern "C" fn must be caught, not unwind
    /// across the FFI boundary. poll_token on a null engine should return a
    /// clean error result rather than panicking the process.
    #[test]
    fn poll_token_on_null_engine_is_a_clean_error_not_a_panic() {
        let result = unsafe { poll_token(std::ptr::null_mut(), 42) };
        assert!(result.is_error);
        assert!(result.text.is_null());
    }
}
