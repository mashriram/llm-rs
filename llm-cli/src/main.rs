use std::net::SocketAddr;
use std::sync::Arc;
use clap::Parser;
use tracing::info;

use llm_scheduler::engine::ServingEngine;
use llm_cli::{
    create_router_with_body_limit, log_compiled_backend_support, load_candle_backend,
    resolve_tokenizer_path, AppState, DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_TOKENS_LIMIT,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Path to the model file (GGUF) or model directory (safetensors).
    #[arg(long)]
    model_path: String,

    /// Path to tokenizer.json. If omitted, looked for next to --model-path
    /// (or inside it, if it's a directory); if not found there either, the
    /// server refuses to start with an actionable error.
    #[arg(long)]
    tokenizer_path: Option<String>,

    #[arg(long, default_value_t = 1024)]
    block_pool_size: usize,

    /// Reject (400 Bad Request) any chat completion request whose
    /// `max_tokens` exceeds this. Protects against a client tying up engine
    /// resources indefinitely with an unbounded generation length.
    #[arg(long, default_value_t = DEFAULT_MAX_TOKENS_LIMIT)]
    max_tokens_limit: usize,

    /// Reject request bodies larger than this many bytes.
    #[arg(long, default_value_t = DEFAULT_MAX_BODY_BYTES)]
    max_body_bytes: usize,

    /// Explicit dequantize mode (slower, uses F16 weights).
    #[arg(long, default_value_t = false)]
    explicit_dequantize: bool,

    /// Force embedding tables to VRAM (GPU) instead of CPU system RAM.
    #[arg(long, default_value_t = false)]
    use_vram_embeddings: bool,

    /// Explicit path to multimodal projector (mmproj) GGUF file
    #[arg(long)]
    mmproj_path: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    info!("Starting llm-cli with args: {:?}", args);
    log_compiled_backend_support();

    // Security posture disclosure (audit finding #4c): this release has no
    // authentication and no rate-limiting. Say so loudly at startup instead
    // of silently having zero protection with no acknowledgment.
    tracing::warn!(
        "no authentication/rate-limiting configured on this server; do not expose it \
         directly to an untrusted network. Only max_tokens ceiling ({}) and request body \
         size ({} bytes) are enforced.",
        args.max_tokens_limit,
        args.max_body_bytes
    );

    let path = std::path::Path::new(&args.model_path);
    if !path.exists() {
        anyhow::bail!(
            "model file not found at {}: pass a valid --model-path",
            args.model_path
        );
    }

    info!("Loading weights from {}...", args.model_path);
    let (backend, meta) = load_candle_backend(
        path,
        args.explicit_dequantize,
        args.use_vram_embeddings,
        args.mmproj_path.map(std::path::PathBuf::from),
    )?;
    info!(
        "Weights loaded successfully. arch: {} | hidden: {} | layers: {}",
        meta.arch, meta.hidden_dim, meta.n_layers
    );

    // Initialize the serving engine
    let engine = Arc::new(ServingEngine::new(backend, args.block_pool_size));
    let model_name = path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("llama-model")
        .to_string();

    info!("Loading tokenizer...");
    let tokenizer_path = match args.tokenizer_path {
        Some(p) => std::path::PathBuf::from(p),
        None => resolve_tokenizer_path(path)?,
    };
    if !tokenizer_path.exists() {
        anyhow::bail!(
            "tokenizer.json not found at {}: pass --tokenizer-path explicitly with a valid path",
            tokenizer_path.display()
        );
    }
    let tokenizer = Arc::new(
        llm_core::tokenizer::LlmTokenizer::from_file(&tokenizer_path).map_err(|e| {
            anyhow::anyhow!(
                "failed to load tokenizer from {}: {}",
                tokenizer_path.display(),
                e
            )
        })?,
    );
    info!("Tokenizer loaded successfully from {}.", tokenizer_path.display());

    let state = Arc::new(AppState {
        engine,
        model_name,
        tokenizer,
        meta: Arc::new(meta),
        max_tokens_limit: args.max_tokens_limit,
    });

    let app = create_router_with_body_limit(state, args.max_body_bytes);

    let addr = format!("{}:{}", args.host, args.port).parse::<SocketAddr>().map_err(|e| {
        anyhow::anyhow!("invalid --host/--port combination ({}:{}): {}", args.host, args.port, e)
    })?;
    info!("Server listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| {
        anyhow::anyhow!("failed to bind {}: {} (is the port already in use?)", addr, e)
    })?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shut down cleanly.");
    Ok(())
}

/// Resolves once Ctrl-C (or, on Unix, SIGTERM) is received, letting
/// `axum::serve`'s graceful shutdown drain in-flight requests instead of
/// killing the process mid-request.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to install SIGTERM handler: {}", e);
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    tracing::info!("shutdown signal received, draining in-flight requests...");
}
