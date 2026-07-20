use std::sync::Arc;
use std::time::Instant;
use std::path::PathBuf;
use std::io::{self, Write, BufRead};
use llm_core::types::{InferRequest, SampleParams};
use llm_scheduler::engine::ServingEngine;
use clap::Parser;
use llm_cli::{
    load_candle_backend, log_compiled_backend_support, maybe_prepend_bos, render_prompt,
    resolve_eos_token_ids, ChatMessage,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactive multi-turn chat CLI with multimodal support")]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model_path: String,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer_path: String,

    /// Sampling temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-p nucleus sampling
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Maximum new tokens per response
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,

    /// Repetition penalty
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Explicit dequantize mode (slower, uses F16 weights)
    #[arg(long, default_value_t = false)]
    explicit_dequantize: bool,

    /// Force embedding tables to VRAM (GPU) instead of CPU system RAM
    #[arg(long, default_value_t = false)]
    use_vram_embeddings: bool,

    /// Explicit path to multimodal projector (mmproj) GGUF file
    #[arg(long)]
    mmproj_path: Option<PathBuf>,

    /// System prompt (optional)
    #[arg(long)]
    system_prompt: Option<String>,
}

// Chat-template rendering (render_prompt/try_jinja_render/manual_format) and
// Gemma BOS handling (maybe_prepend_bos) now live in llm_cli (lib.rs) as the
// single shared implementation also used by the HTTP server's
// chat_completions handler — see audit findings #7/#9/#10.

/// Parse a line into a content string, possibly with an image or audio command.
/// Returns (text_content, image_path, audio_path)
fn parse_user_input(line: &str) -> (String, Option<PathBuf>, Option<PathBuf>) {
    let mut image_path = None;
    let mut audio_path = None;
    let mut text_parts: Vec<&str> = Vec::new();

    for part in line.split_whitespace().collect::<Vec<_>>().windows(1).map(|w| w[0]) {
        text_parts.push(part);
    }

    // Actually let's parse properly line by line
    let mut remaining = line.trim();
    let mut rebuilt_text = String::new();

    while !remaining.is_empty() {
        if let Some(rest) = remaining.strip_prefix("/image ") {
            let mut path_part = String::new();
            let mut after = "";
            let parts: Vec<&str> = rest.split_whitespace().collect();
            
            if rest.starts_with('"') {
                if let Some(end_quote_idx) = rest[1..].find('"') {
                    path_part = rest[1..end_quote_idx + 1].to_string();
                    after = &rest[end_quote_idx + 2..];
                }
            } else if rest.starts_with('\'') {
                if let Some(end_quote_idx) = rest[1..].find('\'') {
                    path_part = rest[1..end_quote_idx + 1].to_string();
                    after = &rest[end_quote_idx + 2..];
                }
            }
            
            if path_part.is_empty() {
                let mut found_path = false;
                for i in 1..=parts.len() {
                    let candidate = parts[0..i].join(" ");
                    let p = PathBuf::from(&candidate);
                    if p.exists() {
                        path_part = candidate;
                        after = rest.strip_prefix(&path_part).unwrap_or("").trim();
                        found_path = true;
                        break;
                    }
                }
                if !found_path && !parts.is_empty() {
                    path_part = parts[0].to_string();
                    after = rest.strip_prefix(&path_part).unwrap_or("").trim();
                }
            }

            let path = PathBuf::from(path_part.trim());
            if path.exists() {
                image_path = Some(path);
                rebuilt_text.push_str("<image> ");
            } else {
                eprintln!("[chat] Image not found: {}", path_part.trim());
            }
            remaining = after.trim();
        } else if let Some(rest) = remaining.strip_prefix("/audio ") {
            let mut path_part = String::new();
            let mut after = "";
            let parts: Vec<&str> = rest.split_whitespace().collect();
            
            if rest.starts_with('"') {
                if let Some(end_quote_idx) = rest[1..].find('"') {
                    path_part = rest[1..end_quote_idx + 1].to_string();
                    after = &rest[end_quote_idx + 2..];
                }
            } else if rest.starts_with('\'') {
                if let Some(end_quote_idx) = rest[1..].find('\'') {
                    path_part = rest[1..end_quote_idx + 1].to_string();
                    after = &rest[end_quote_idx + 2..];
                }
            }
            
            if path_part.is_empty() {
                let mut found_path = false;
                for i in 1..=parts.len() {
                    let candidate = parts[0..i].join(" ");
                    let p = PathBuf::from(&candidate);
                    if p.exists() {
                        path_part = candidate;
                        after = rest.strip_prefix(&path_part).unwrap_or("").trim();
                        found_path = true;
                        break;
                    }
                }
                if !found_path && !parts.is_empty() {
                    path_part = parts[0].to_string();
                    after = rest.strip_prefix(&path_part).unwrap_or("").trim();
                }
            }

            let path = PathBuf::from(path_part.trim());
            if path.exists() {
                audio_path = Some(path);
                rebuilt_text.push_str("<audio> ");
            } else {
                eprintln!("[chat] Audio not found: {}", path_part.trim());
            }
            remaining = after.trim();
        } else {
            rebuilt_text.push_str(remaining);
            break;
        }
    }

    (rebuilt_text.trim().to_string(), image_path, audio_path)
}

fn print_help() {
    println!("\n  Commands:");
    println!("    /clear        - Clear conversation history");
    println!("    /image <path> - Include an image in your next message");
    println!("    /audio <path> - Include an audio file in your next message");
    println!("    /help         - Show this help");
    println!("    /exit, /quit  - Exit chat\n");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    log_compiled_backend_support();

    // Validate arguments up front with actionable errors rather than
    // failing deep inside model/tokenizer loading with a raw io::Error.
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

    println!("══════════════════════════════════════════════");
    println!("  llm-rs Interactive Chat");
    println!("══════════════════════════════════════════════");
    println!("  Model:     {}", args.model_path);
    println!("  Tokenizer: {}", args.tokenizer_path);
    println!("  Temp: {}  Top-p: {}  Max tokens: {}", args.temperature, args.top_p, args.max_new_tokens);
    println!("  Type /help for commands");
    println!("══════════════════════════════════════════════\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer = Arc::new(
        llm_core::tokenizer::LlmTokenizer::from_file(&args.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {}: {}", args.tokenizer_path, e))?,
    );

    // Load model
    println!("Loading model (this may take a moment)...");
    let load_start = Instant::now();
    let (backend, meta) = load_candle_backend(
        model_path,
        args.explicit_dequantize,
        args.use_vram_embeddings,
        args.mmproj_path,
    )?;
    println!(
        "Model loaded in {:.2?} | arch: {} | hidden: {} | layers: {} | heads: {} | vocab: {}",
        load_start.elapsed(),
        meta.arch,
        meta.hidden_dim,
        meta.n_layers,
        meta.n_heads,
        meta.vocab_size
    );
    if meta.has_vision_encoder {
        println!("  Vision encoder: YES (image inputs supported via /image <path>)");
    }
    if meta.chat_template.is_some() {
        println!("  Chat template:  Jinja loaded from GGUF ✓");
    } else {
        println!("  Chat template:  using fallback format for arch '{}'", meta.arch);
    }
    println!();

    let eos_token_ids = resolve_eos_token_ids(backend.as_ref(), &tokenizer);
    let engine = Arc::new(ServingEngine::new(backend, 2048));
    let meta = Arc::new(meta);

    let mut conversation: Vec<ChatMessage> = Vec::new();
    let mut seq_id: u64 = 1;

    // Add system prompt if provided
    if let Some(sys) = &args.system_prompt {
        conversation.push(ChatMessage { role: "system".to_string(), content: sys.clone() });
    }

    let stdin = io::stdin();
    loop {
        // Print prompt
        print!("You: ");
        io::stdout().flush()?;

        let mut line = String::new();
        let n = stdin.lock().read_line(&mut line)?;
        if n == 0 {
            // EOF
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Handle commands
        match line {
            "/exit" | "/quit" => {
                println!("Goodbye!");
                break;
            }
            "/clear" => {
                conversation.clear();
                if let Some(sys) = &args.system_prompt {
                    conversation.push(ChatMessage { role: "system".to_string(), content: sys.clone() });
                }
                *llm_core::backends::ACTIVE_IMAGE_PATH.lock() = None;
                *llm_core::backends::ACTIVE_AUDIO_PATH.lock() = None;
                println!("[chat] Conversation cleared.\n");
                continue;
            }
            "/help" => {
                print_help();
                continue;
            }
            _ => {}
        }

        // Parse text + optional media paths
        let (text_content, image_path, audio_path) = parse_user_input(line);

        if let Some(ref img) = image_path {
            *llm_core::backends::ACTIVE_IMAGE_PATH.lock() = Some(img.to_string_lossy().to_string());
        }

        if let Some(ref aud) = audio_path {
            *llm_core::backends::ACTIVE_AUDIO_PATH.lock() = Some(aud.to_string_lossy().to_string());
        }

        if text_content.is_empty() && image_path.is_none() && audio_path.is_none() {
            continue;
        }

        // Add user message to history
        let mut msg_content = text_content.clone();
        if image_path.is_some() && !msg_content.contains("<image>") && !msg_content.contains("<|image|>") {
            msg_content = format!("<image>\n{}", msg_content);
        }
        if audio_path.is_some() && !msg_content.contains("<audio>") && !msg_content.contains("<|audio|>") {
            msg_content = format!("<audio>\n{}", msg_content);
        }
        conversation.push(ChatMessage { role: "user".to_string(), content: msg_content });

        // Render the full conversation to tokens
        let mut prompt_str = render_prompt(&conversation, &meta);
        if meta.has_vision_encoder {
            // Dynamically resolve the image pad token name from the tokenizer
            let pad_token = if tokenizer.token_to_id("<|image_pad|>").is_some() {
                "<|image_pad|>"
            } else if tokenizer.token_to_id("<|image|>").is_some() {
                "<|image|>"
            } else if tokenizer.token_to_id("<image>").is_some() {
                "<image>"
            } else {
                "<|image|>" // Fallback
            };

            let (start_token, end_token) = if tokenizer.token_to_id("<|vision_start|>").is_some()
                && tokenizer.token_to_id("<|vision_end|>").is_some() {
                (Some("<|vision_start|>"), Some("<|vision_end|>"))
            } else if tokenizer.token_to_id("<|image>").is_some()
                && tokenizer.token_to_id("<image|>").is_some() {
                (Some("<|image>"), Some("<image|>"))
            } else {
                (None, None)
            };

            // Compute number of image tokens dynamically from model architecture metadata.
            // patches_per_side = image_size / patch_size  →  total_patches = patches_per_side^2
            // spatial_merge reduces tokens: total / (merge_size^2), defaulting to 1 (no merge).
            let image_size = meta.vision_image_size.unwrap_or(224);
            let patch_size = meta.vision_patch_size.unwrap_or(16);
            let merge = meta.spatial_merge_size.unwrap_or(1).max(1);
            let patches_per_side = image_size / patch_size.max(1);
            let num_image_tokens = (patches_per_side * patches_per_side) / (merge * merge);
            let num_image_tokens = num_image_tokens.max(16); // always at least 16 so run-based splice triggers

            let pads = pad_token.repeat(num_image_tokens);
            let replacement = match (start_token, end_token) {
                (Some(st), Some(et)) => format!("{st}{pads}{et}"),
                _ => pads,
            };
            prompt_str = prompt_str.replace("<image>", &replacement);
        }
        if meta.has_audio_encoder {
            // Dynamically resolve the audio pad token name from the tokenizer
            let pad_token = if tokenizer.token_to_id("<|audio_pad|>").is_some() {
                "<|audio_pad|>"
            } else if tokenizer.token_to_id("<|audio|>").is_some() {
                "<|audio|>"
            } else if tokenizer.token_to_id("<audio>").is_some() {
                "<audio>"
            } else {
                "<|audio|>" // Fallback
            };

            // `audio_embedding_length` is the encoder's HIDDEN dimension (e.g.
            // 1024 for Gemma-Conformer), not a token count - using it directly
            // as a placeholder-token count was a real bug (confirmed via a real
            // audio file: it produced a splice-length-mismatch crash, since the
            // encoder's actual output length has nothing to do with its hidden
            // size). `load_audio`'s mel pipeline always produces a fixed 3000
            // frames, so each architecture's output token count is a fixed
            // constant determined only by its conv-subsampling factor: 4x for
            // Gemma-Conformer (2 stride-2 convs -> 750), 2x for Whisper (one
            // stride-1 + one stride-2 conv -> 1500). `audio_num_mel_bins`
            // already tells us which architecture this is (128 vs 80).
            let num_audio_tokens = match meta.audio_num_mel_bins {
                Some(80) => 1500,  // Whisper: 3000 / 2
                _ => 750,          // Gemma-Conformer (128 mel bins): 3000 / 4
            };
            let pads = pad_token.repeat(num_audio_tokens);
            prompt_str = prompt_str.replace("<audio>", &pads);
        }
        let mut prompt_tokens = tokenizer.encode(&prompt_str, true)?;
        maybe_prepend_bos(&mut prompt_tokens, &meta);
        let prompt_len = prompt_tokens.len();

        let req = InferRequest {
            seq_id,
            prompt_tokens,
            max_new_tokens: args.max_new_tokens,
            sample_params: SampleParams {
                temperature: args.temperature,
                top_p: args.top_p,
                top_k: 0,
                repetition_penalty: args.repetition_penalty,
                max_new_tokens: args.max_new_tokens,
            },
        };

        // Subscribe to events BEFORE adding the request
        let mut rx = engine.subscribe();
        let this_seq_id = seq_id;
        seq_id += 1;

        print!("Assistant: ");
        io::stdout().flush()?;

        engine.add_request(req)?;

        let mut assistant_response = String::new();
        let gen_result = llm_cli::consume_generation(
            &mut rx,
            this_seq_id,
            &tokenizer,
            &eos_token_ids,
            args.max_new_tokens,
            |_token_id, text| {
                if !text.is_empty() {
                    print!("{}", text);
                    let _ = io::stdout().flush();
                    assistant_response.push_str(text);
                }
            },
        ).await;

        println!(); // newline after assistant response

        // Print timing stats
        let n_gen = gen_result.tokens.len();
        if let Some(ttft_dur) = gen_result.ttft {
            let prefill_tps = if ttft_dur.as_secs_f64() > 0.0 { prompt_len as f64 / ttft_dur.as_secs_f64() } else { 0.0 };
            let decode_dur = gen_result.total_dur.saturating_sub(ttft_dur);
            let decode_tps = if n_gen > 1 && decode_dur.as_secs_f64() > 0.0 { (n_gen - 1) as f64 / decode_dur.as_secs_f64() } else { 0.0 };
            println!(
                "  [stats] TTFT: {:.0}ms | Prefill: {:.1} t/s | Decode: {:.1} t/s | Tokens: {}",
                ttft_dur.as_millis(), prefill_tps, decode_tps, n_gen
            );
        }
        if gen_result.lagged {
            println!("  [warn] event stream lagged; response above may be truncated.");
        }
        println!();

        // Add assistant turn to conversation history (strip EOS tokens if present)
        let _clean_response = assistant_response.trim_end_matches(|c: char| c == '<' || c == '>' || c.is_alphanumeric()).trim().to_string();
        if !assistant_response.is_empty() {
            conversation.push(ChatMessage { role: "assistant".to_string(), content: assistant_response.clone() });
        }

        // Trim conversation to avoid exceeding context length
        // Keep system prompt + last N turns
        let max_turns = 20;
        let start_idx = if conversation.len() > max_turns * 2 + 1 {
            let sys_offset = if conversation.first().map(|m| m.role == "system").unwrap_or(false) { 1 } else { 0 };
            conversation.len() - max_turns * 2 - sys_offset
        } else {
            0
        };
        if start_idx > 0 {
            let sys = if conversation[0].role == "system" { Some(conversation[0].clone()) } else { None };
            conversation = conversation[start_idx..].to_vec();
            if let Some(s) = sys {
                if conversation.first().map(|m| m.role != "system").unwrap_or(true) {
                    conversation.insert(0, s);
                }
            }
        }
    }

    Ok(())
}
