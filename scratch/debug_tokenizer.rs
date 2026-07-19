use tokenizers::Tokenizer;

fn test_tok(name: &str, tok_path: &str, strings: &[&str]) -> anyhow::Result<()> {
    println!("=== {} ===", name);
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    for s in strings {
        let enc = tokenizer.encode(s.to_string(), true).map_err(|e| anyhow::anyhow!("{:?}", e))?;
        println!("Tokenizing '{}': IDs={:?}, Tokens={:?}", s, enc.get_ids(), enc.get_tokens());
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    test_tok("Gemma 4", "/home/shri/.cache/llm-rs/hub/google--gemma-4-E2B-it/tokenizer.json", &[
        "<|turn>", "<turn|>", "<|turn>user\n", "<|turn>model\n", "<image>", "<|image|>", "<|image_pad|>"
    ])?;

    test_tok("Qwen 2.5", "/home/shri/.cache/llm-rs/hub/Qwen--Qwen2.5-1.5B-Instruct/tokenizer.json", &[
        "<|im_start|>", "<|im_end|>", "<|im_start|>user\n", "<|im_start|>assistant\n"
    ])?;

    test_tok("SmolLM3", "/home/shri/.cache/llm-rs/hub/HuggingFaceTB--SmolLM3-3B/tokenizer.json", &[
        "<|im_start|>", "<|im_end|>", "<|im_start|>user\n", "<|im_start|>assistant\n"
    ])?;

    Ok(())
}

