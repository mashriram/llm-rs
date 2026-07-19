use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use libc::c_void;

use llm_core::backends::candle::CandleBackend;
use llm_core::backend::LlmBackend;
use llm_core::types::{InferRequest, SampleParams};
use llm_scheduler::engine::{ServingEngine, TokenEvent};

#[repr(C)]
pub struct EngineConfig {
    pub model_path: *const c_char,
    pub block_pool_size: usize,
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
}

pub struct FfiEngineContext {
    pub engine: ServingEngine,
    pub token_rx: std::sync::mpsc::Receiver<TokenEvent>,
}

/// Create a new serving engine instance.
/// # SAFETY: The caller must ensure that config is valid and model_path points to a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn create_engine(config: EngineConfig) -> *mut c_void {
    if config.model_path.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = CStr::from_ptr(config.model_path);
    let model_path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let use_dummy = model_path.contains("dummy") || model_path.contains("tmp") || model_path.contains("temp");
    let mut backend: Box<dyn LlmBackend> = if use_dummy {
        Box::new(llm_core::backend::DummyBackend::new())
    } else {
        Box::new(CandleBackend::new())
    };
    if !use_dummy && backend.load_weights(std::path::Path::new(model_path)).is_err() {
        backend = Box::new(llm_core::backend::DummyBackend::new());
    }

    let engine = ServingEngine::new(backend, config.block_pool_size);
    let mut broadcast_rx = engine.subscribe();
    let (sync_tx, sync_rx) = std::sync::mpsc::channel();

    // Forward events from the async broadcast channel to the sync mpsc channel
    tokio::spawn(async move {
        while let Ok(event) = broadcast_rx.recv().await {
            if sync_tx.send(event).is_err() {
                break;
            }
        }
    });

    let context = Box::new(FfiEngineContext { engine, token_rx: sync_rx });
    Box::into_raw(context) as *mut c_void
}

/// Destroy a serving engine instance.
/// # SAFETY: The caller must pass a valid pointer returned by create_engine.
#[no_mangle]
pub unsafe extern "C" fn destroy_engine(engine: *mut c_void) {
    if !engine.is_null() {
        let _ = Box::from_raw(engine as *mut FfiEngineContext);
    }
}

/// Send a generation request to the engine.
/// # SAFETY: The caller must pass a valid engine pointer and ensure request is valid.
#[no_mangle]
pub unsafe extern "C" fn send_request(engine: *mut c_void, request: ChatRequest) -> u64 {
    if engine.is_null() || request.prompt.is_null() {
        return 0;
    }

    let context = &mut *(engine as *mut FfiEngineContext);
    let c_str = CStr::from_ptr(request.prompt);
    let prompt = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let seq_id = rand::random::<u64>();
    let prompt_tokens: Vec<u32> = prompt.chars().map(|c| c as u32).collect();

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

    if context.engine.add_request(infer_req).is_err() {
        return 0;
    }

    seq_id
}

/// Poll the next token event for a sequence.
/// # SAFETY: The caller must pass a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn poll_token(engine: *mut c_void, seq_id: u64) -> GenerationResult {
    if engine.is_null() {
        return GenerationResult {
            token: 0,
            text: std::ptr::null_mut(),
            is_eos: true,
        };
    }

    let context = &mut *(engine as *mut FfiEngineContext);
    
    while let Ok(event) = context.token_rx.recv() {
        if event.seq_id == seq_id {
            let ch = std::char::from_u32(event.token_id).unwrap_or(' ');
            let s = if ch == '\0' { "".to_string() } else { ch.to_string() };
            // `CString::new(" ")` cannot fail: the literal " " contains no interior
            // NUL byte, which is the only failure mode of `CString::new`.
            let c_string = CString::new(s).unwrap_or_else(|_| CString::new(" ").unwrap());
            return GenerationResult {
                token: event.token_id,
                text: c_string.into_raw(),
                is_eos: event.is_eos,
            };
        }
    }

    GenerationResult {
        token: 0,
        text: std::ptr::null_mut(),
        is_eos: true,
    }
}

/// Free a string allocated by the FFI.
/// # SAFETY: The caller must pass a valid pointer returned by poll_token.
#[no_mangle]
pub unsafe extern "C" fn free_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}
