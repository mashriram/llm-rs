use std::sync::Arc;
use std::time::Instant;
use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_scheduler::engine::ServingEngine;
use llm_core::types::{InferRequest, SampleParams};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let model_path = "/home/mukundan/learning/llm/qwen3-vl-4b-instruct.Q4_K_M.gguf";
    let tokenizer_path = "/home/mukundan/learning/llm/llm-rs/qwen2.5-0.5b/tokenizer.json";
    
    println!("Loading tokenizer from {}...", tokenizer_path);
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(tokenizer_path)?);
    
    println!("Loading model from {}...", model_path);
    let mut backend = Box::new(CandleBackend::new());
    if std::env::var("LLM_EXPLICIT_DEQUANTIZE").is_ok() {
        backend.set_explicit_dequantize(true);
    }
    let start_load = Instant::now();
    let meta = backend.load_weights(std::path::Path::new(model_path))?;
    println!("Model loaded in {:.2?} (vocab_size: {}, hidden_dim: {})", start_load.elapsed(), meta.vocab_size, meta.hidden_dim);
    
    let engine = Arc::new(ServingEngine::new(backend, 1024));
    let mut rx = engine.subscribe();
    
    let raw_prompt = "Explain the theory of relativity in 2 sentences.";
    let prompt = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", raw_prompt);
    let prompt_tokens = tokenizer.encode(&prompt, true)?;
    let prompt_len = prompt_tokens.len();
    println!("Prompt ({} tokens): \"{}\"", prompt_len, raw_prompt);
    
    let req = InferRequest {
        seq_id: 1,
        prompt_tokens,
        max_new_tokens: 50,
        sample_params: SampleParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
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
            print!("[ID: {}, String: {:?}]", event.token_id, token_str);
            std::io::Write::flush(&mut std::io::stdout())?;
            
            generated_tokens.push(event.token_id);
            last_token_time = now;
            
            if event.is_eos {
                break;
            }
        }
    }
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
