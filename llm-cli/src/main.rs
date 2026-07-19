use std::net::SocketAddr;
use std::sync::Arc;
use clap::Parser;
use tracing::info;

use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_scheduler::engine::ServingEngine;
use llm_cli::{create_router, AppState};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long)]
    model_path: String,

    #[arg(long, default_value_t = 1024)]
    block_pool_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    info!("Starting llm-cli with args: {:?}", args);

    // Initialize the Candle reference backend
    let mut backend = Box::new(CandleBackend::new());
    info!("Loading weights from {}...", args.model_path);
    let path = std::path::Path::new(&args.model_path);
    let _meta = backend.load_weights(path)?;
    info!("Weights loaded successfully.");

    // Initialize the serving engine
    let engine = Arc::new(ServingEngine::new(backend, args.block_pool_size));
    let model_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("llama-model")
        .to_string();

    info!("Loading tokenizer...");
    let tokenizer_path = if path.is_file() {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("model_path {:?} has no parent directory to look for a tokenizer in", path))?;
        let same_dir = parent.join("tokenizer.json");
        if same_dir.exists() {
            same_dir
        } else {
            let fallback_05b = parent.join("llm-rs/qwen2.5-0.5b/tokenizer.json");
            let fallback_15b = parent.join("llm-rs/qwen2.5-1.5b/tokenizer.json");
            if fallback_05b.exists() {
                fallback_05b
            } else if fallback_15b.exists() {
                fallback_15b
            } else {
                let rel_05b = std::path::Path::new("qwen2.5-0.5b/tokenizer.json");
                let rel_15b = std::path::Path::new("qwen2.5-1.5b/tokenizer.json");
                if rel_05b.exists() {
                    rel_05b.to_path_buf()
                } else if rel_15b.exists() {
                    rel_15b.to_path_buf()
                } else {
                    parent.join("tokenizer.json")
                }
            }
        }
    } else {
        path.join("tokenizer.json")
    };
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(tokenizer_path)?);
    info!("Tokenizer loaded successfully.");

    let state = Arc::new(AppState {
        engine,
        model_name,
        tokenizer,
    });

    let app = create_router(state);

    let addr = format!("{}:{}", args.host, args.port).parse::<SocketAddr>()?;
    info!("Server listening on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
