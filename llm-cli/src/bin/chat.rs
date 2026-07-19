use std::sync::Arc;
use std::time::Instant;
use std::path::PathBuf;
use std::io::{self, Write, BufRead};
use llm_core::backend::LlmBackend;
use llm_core::backends::candle::CandleBackend;
use llm_core::types::{InferRequest, SampleParams, ModelMeta};
use llm_scheduler::engine::ServingEngine;
use clap::Parser;
use minijinja::{Environment, Value, context};
use minijinja_contrib::add_to_environment;

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

    /// System prompt (optional)
    #[arg(long)]
    system_prompt: Option<String>,
}

/// A single turn in the conversation
#[derive(Debug, Clone)]
struct Message {
    role: String,
    content: String,
}

/// Detect the chat format from ModelMeta and try to render via Jinja, with a robust fallback
fn render_prompt(messages: &[Message], meta: &ModelMeta) -> String {
    // Try Jinja rendering from embedded chat_template first
    if let Some(template_str) = &meta.chat_template {
        match try_jinja_render(template_str, messages, meta) {
            Ok(rendered) => return rendered,
            Err(e) => {
                eprintln!("[chat] Jinja render failed ({}), falling back to manual format", e);
            }
        }
    }
    // Manual fallback based on architecture
    manual_format(messages, meta)
}

fn try_jinja_render(template_str: &str, messages: &[Message], _meta: &ModelMeta) -> anyhow::Result<String> {
    let mut env = Environment::new();
    // Add Python-compat builtins (tojson, etc.)
    add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    // Register strftime_now: returns today's date string (HF templates use this)
    env.add_function("strftime_now", |fmt: &str| -> String {
        // Return a plausible static date; no std::time formatting in minijinja natively
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        // Basic date from seconds (approx)
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

    // Convert messages to minijinja-compatible values
    let msgs: Vec<Value> = messages.iter().map(|m| {
        context! {
            role => m.role.as_str(),
            content => m.content.as_str(),
        }
    }).collect();

    env.add_template("chat", &cleaned)?;
    let tmpl = env.get_template("chat")?;

    let ctx = context! {
        messages => msgs,
        add_generation_prompt => true,
        enable_thinking => false,
    };

    let rendered = tmpl.render(ctx)?;
    Ok(rendered)
}

fn manual_format(messages: &[Message], meta: &ModelMeta) -> String {
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

/// Insert BOS token for Gemma if needed
fn maybe_prepend_bos(tokens: &mut Vec<u32>, meta: &ModelMeta) {
    if meta.is_gemma && (tokens.is_empty() || tokens[0] != 2) {
        tokens.insert(0, 2);
    }
}

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
    let tokenizer = Arc::new(llm_core::tokenizer::LlmTokenizer::from_file(&args.tokenizer_path)?);

    // Load model
    println!("Loading model (this may take a moment)...");
    let load_start = Instant::now();
    let mut backend = Box::new(CandleBackend::new());
    if args.explicit_dequantize {
        backend.set_explicit_dequantize(true);
    }
    if args.use_vram_embeddings {
        backend.set_use_vram_embeddings(true);
    }
    let meta = backend.load_weights(std::path::Path::new(&args.model_path))?;
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

    let mut eos_token_ids: Vec<u32> = Vec::new();
    if let Some(id) = backend.eos_token_id() {
        eos_token_ids.push(id);
    } else {
        eprintln!("Warning: no EOS token id could be determined for this model; relying on --max-new-tokens and known chat-template stop strings only.");
    }
    for tok in &["<|im_end|>", "<end_of_turn>", "<turn|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_token_ids.push(id);
        }
    }
    let engine = Arc::new(ServingEngine::new(backend, 2048));
    let meta = Arc::new(meta);

    let mut conversation: Vec<Message> = Vec::new();
    let mut seq_id: u64 = 1;

    // Add system prompt if provided
    if let Some(sys) = &args.system_prompt {
        conversation.push(Message { role: "system".to_string(), content: sys.clone() });
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
                    conversation.push(Message { role: "system".to_string(), content: sys.clone() });
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
        conversation.push(Message { role: "user".to_string(), content: text_content.clone() });

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

            let num_audio_tokens = meta.audio_embedding_length.unwrap_or(750);
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

        let gen_start = Instant::now();
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut assistant_response = String::new();
        let mut ttft: Option<std::time::Duration> = None;

        loop {
            match rx.recv().await {
                Ok(event) if event.seq_id == this_seq_id => {
                    if ttft.is_none() {
                        ttft = Some(gen_start.elapsed());
                    }

                    // Decode and print token
                    let token_str = tokenizer.decode(&[event.token_id], true).unwrap_or_default();
                    if !token_str.is_empty() {
                        print!("{}", token_str);
                        io::stdout().flush()?;
                        assistant_response.push_str(&token_str);
                    }
                    generated_tokens.push(event.token_id);

                    if event.is_eos || eos_token_ids.contains(&event.token_id) {
                        break;
                    }
                    if generated_tokens.len() >= args.max_new_tokens {
                        break;
                    }
                }
                Ok(_) => continue, // Different seq_id
                Err(_) => break,
            }
        }

        println!(); // newline after assistant response

        // Print timing stats
        let total_dur = gen_start.elapsed();
        let n_gen = generated_tokens.len();
        if let Some(ttft_dur) = ttft {
            let prefill_tps = prompt_len as f64 / ttft_dur.as_secs_f64();
            let decode_dur = total_dur.saturating_sub(ttft_dur);
            let decode_tps = if n_gen > 1 { (n_gen - 1) as f64 / decode_dur.as_secs_f64() } else { 0.0 };
            println!(
                "  [stats] TTFT: {:.0}ms | Prefill: {:.1} t/s | Decode: {:.1} t/s | Tokens: {}",
                ttft_dur.as_millis(), prefill_tps, decode_tps, n_gen
            );
        }
        println!();

        // Add assistant turn to conversation history (strip EOS tokens if present)
        let _clean_response = assistant_response.trim_end_matches(|c: char| c == '<' || c == '>' || c.is_alphanumeric()).trim().to_string();
        if !assistant_response.is_empty() {
            conversation.push(Message { role: "assistant".to_string(), content: assistant_response.clone() });
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
