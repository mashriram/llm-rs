use std::sync::Arc;
use std::time::Instant;
use llm_scheduler::engine::ServingEngine;
use llm_core::types::{InferRequest, SampleParams};
use clap::Parser;
use llm_cli::{
    consume_generation, load_candle_backend, log_compiled_backend_support, maybe_prepend_bos,
    print_gen_stats, resolve_eos_token_ids,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark llm-rs inference throughput")]
struct Args {
    #[arg(long)]
    model_path: String,

    #[arg(long)]
    tokenizer_path: String,

    #[arg(long, default_value = "Explain the theory of relativity in 2 sentences.")]
    prompt: String,

    #[arg(long, default_value_t = 50)]
    max_new_tokens: usize,

    #[arg(long, default_value_t = false)]
    explicit_dequantize: bool,

    /// Explicit path to multimodal projector (mmproj) GGUF file
    #[arg(long)]
    mmproj_path: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    log_compiled_backend_support();

    let model_path = std::path::Path::new(&args.model_path);
    if !model_path.exists() {
        anyhow::bail!("model file not found at {}: pass a valid --model-path", args.model_path);
    }
    let tokenizer_path = std::path::Path::new(&args.tokenizer_path);
    if !tokenizer_path.exists() {
        anyhow::bail!(
            "tokenizer.json not found at {}: pass --tokenizer-path explicitly with a valid path",
            args.tokenizer_path
        );
    }

    println!("Loading tokenizer from {}...", args.tokenizer_path);
    let tokenizer = Arc::new(
        llm_core::tokenizer::LlmTokenizer::from_file(&args.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {}: {}", args.tokenizer_path, e))?,
    );

    println!("Loading model from {}...", args.model_path);
    let start_load = Instant::now();
    // benchmark_speed previously never called set_use_vram_embeddings; kept
    // as `false` here to preserve prior behavior (no CLI flag existed for it).
    let (backend, meta) = load_candle_backend(
        model_path,
        args.explicit_dequantize,
        false,
        args.mmproj_path.map(std::path::PathBuf::from),
    )?;
    println!(
        "Model loaded in {:.2?} (vocab_size: {}, hidden_dim: {})",
        start_load.elapsed(),
        meta.vocab_size,
        meta.hidden_dim
    );

    let eos_token_ids = resolve_eos_token_ids(backend.as_ref(), &tokenizer);
    let engine = Arc::new(ServingEngine::new(backend, 1024));
    let mut rx = engine.subscribe();

    let prompt = llm_cli::render_prompt(
        &[llm_cli::ChatMessage {
            role: "user".to_string(),
            content: args.prompt.clone(),
        }],
        &meta,
    );
    let mut prompt_tokens = tokenizer.encode(&prompt, true)?;
    maybe_prepend_bos(&mut prompt_tokens, &meta);
    let prompt_len = prompt_tokens.len();
    println!("Prompt ({} tokens): \"{}\"", prompt_len, args.prompt);

    let req = InferRequest {
        seq_id: 1,
        prompt_tokens,
        max_new_tokens: args.max_new_tokens,
        sample_params: SampleParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            max_new_tokens: args.max_new_tokens,
        },
    };

    println!("Sending inference request...");
    engine.add_request(req)?;

    print!("Assistant: ");
    use std::io::Write;
    std::io::stdout().flush()?;
    let result = consume_generation(&mut rx, 1, &tokenizer, &eos_token_ids, args.max_new_tokens, |token_id, text| {
        if text.is_empty() {
            print!("[ID:{}]", token_id);
        } else {
            print!("{}", text);
        }
        let _ = std::io::stdout().flush();
    }).await;
    println!("\nGenerated token IDs: {:?}", result.tokens);
    println!("\n");

    print_gen_stats(prompt_len, &result);

    Ok(())
}
