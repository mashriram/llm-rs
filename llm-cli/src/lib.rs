use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use axum::{
    routing::post,
    response::sse::{Event, Sse},
    Json, Router, Extension,
};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tracing::error;

use llm_core::types::{InferRequest, SampleParams};
use llm_scheduler::engine::ServingEngine;

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
    pub tokenizer: Arc<llm_core::tokenizer::LlmTokenizer>,
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

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .layer(Extension(state))
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

pub async fn chat_completions(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let seq_id = rand::random::<u64>();

    let mut prompt = String::new();
    for msg in &req.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let prompt_tokens = match state.tokenizer.encode(&prompt, true) {
        Ok(t) => t,
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
        let stream = BroadcastStream::new(token_rx)
            .filter_map(move |res| {
                match res {
                    Ok(event) if event.seq_id == seq_id => {
                        let text = tokenizer.decode(&[event.token_id], true).unwrap_or_else(|_| " ".to_string());
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
                                finish_reason: if event.is_eos { Some("stop".to_string()) } else { None },
                            }],
                        };
                        match serde_json::to_string(&chunk) {
                            Ok(json) => Some(Ok::<Event, std::convert::Infallible>(
                                Event::default().data(json),
                            )),
                            Err(e) => {
                                error!("Failed to serialize chat completion chunk: {:?}", e);
                                None
                            }
                        }
                    }
                    _ => None,
                }
            });

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
                Err(_) => break,
                _ => {}
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
