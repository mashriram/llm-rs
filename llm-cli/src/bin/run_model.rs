use std::sync::Arc;
use std::time::Instant;
use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_scheduler::engine::ServingEngine;
use llm_core::types::{InferRequest, SampleParams};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "/home/mukundan/learning/llm/qwen3-vl-4b-instruct.Q4_K_M.gguf")]
    model_path: String,

    #[arg(long, default_value = "/home/mukundan/learning/llm/llm-rs/qwen2.5-0.5b/tokenizer.json")]
    tokenizer_path: String,

    #[arg(long, default_value = "Explain the theory of relativity in 2 sentences.")]
    prompt: String,

    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    println!("Loading tokenizer from {}...", args.tokenizer_path);
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(&args.tokenizer_path)?);

    println!("Loading model from {}...", args.model_path);
    let mut backend = Box::new(CandleBackend::new());
    let start_load = Instant::now();
    let meta = backend.load_weights(std::path::Path::new(&args.model_path))?;
    println!(
        "Model loaded in {:.2?} (vocab_size: {}, hidden_dim: {})",
        start_load.elapsed(),
        meta.vocab_size,
        meta.hidden_dim
    );

    let engine = Arc::new(ServingEngine::new(backend, 1024));
    let mut rx = engine.subscribe();

    let prompt = if meta.is_gemma {
        format!("<|turn>user\n{}<turn|>\n<|turn>model\n", args.prompt)
    } else {
        format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt)
    };
    let mut prompt_tokens = tokenizer.encode(&prompt, true)?;
    if meta.is_gemma && (prompt_tokens.is_empty() || prompt_tokens[0] != 2) {
        prompt_tokens.insert(0, 2);
    }
    let prompt_len = prompt_tokens.len();
    println!("Prompt ({} tokens): \"{}\"", prompt_len, args.prompt);
    println!("Prompt token IDs: {:?}", prompt_tokens);

    let req = InferRequest {
        seq_id: 1,
        prompt_tokens,
        max_new_tokens: 50,
        sample_params: SampleParams {
            temperature: args.temperature,
            top_p: args.top_p,
            top_k: 0,
            repetition_penalty: args.repetition_penalty,
            max_new_tokens: 50,
        },
    };

    println!("Sending inference request...");
    let start_gen = Instant::now();
    engine.add_request(req)?;

    let mut generated_tokens = Vec::new();
    let mut ttft = None;
    let mut last_token_time = start_gen;
    let mut first_token_time = None;

    while let Ok(event) = rx.recv().await {
        if event.seq_id == 1 {
            let now = Instant::now();
            if ttft.is_none() {
                ttft = Some(now.duration_since(start_gen));
                first_token_time = Some(now);
                print!("Assistant: ");
            }

            let token_str = tokenizer.decode(&[event.token_id], true)?;
            if token_str.is_empty() {
                print!("[ID:{}]", event.token_id);
            } else {
                print!("{}", token_str);
            }
            std::io::Write::flush(&mut std::io::stdout())?;

            generated_tokens.push(event.token_id);
            last_token_time = now;

            if event.is_eos {
                break;
            }
        }
    }
    println!("\nGenerated token IDs: {:?}", generated_tokens);
    println!("\n");

    let total_duration = start_gen.elapsed();
    let num_generated = generated_tokens.len();

    println!("--------------------------------------");
    println!("Benchmark Results (llm-rs):");
    println!("--------------------------------------");
    if let Some(t) = ttft {
        println!("Time to First Token (TTFT): {:.2?}", t);
        println!("Prefill Speed: {:.2} tokens/sec", prompt_len as f64 / t.as_secs_f64());
    }

    if let Some(first_t) = first_token_time {
        let decode_duration = last_token_time.duration_since(first_t);
        println!("Decode Duration: {:.2?}", decode_duration);
        if num_generated > 1 {
            println!("Decode Speed: {:.2} tokens/sec", (num_generated - 1) as f64 / decode_duration.as_secs_f64());
        }
    }

    println!("Total Generation Time: {:.2?}", total_duration);
    println!("Total Tokens Generated: {}", num_generated);
    println!("Overall Speed: {:.2} tokens/sec", num_generated as f64 / total_duration.as_secs_f64());
    println!("--------------------------------------");

    Ok(())
}
