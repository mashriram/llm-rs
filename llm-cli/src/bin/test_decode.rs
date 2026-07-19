//! Dev tool: decode a hardcoded list of token ids with a given tokenizer.
//! Not part of the production CLI surface — low priority for hardening.
fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/gemma4_tokenizer.json".to_string());
    let tokenizer = tokenizers::Tokenizer::from_file(&path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer from {}: {} (pass a valid tokenizer.json path as the first argument)", path, e))?;
    let ids = vec![12766, 3140, 3140, 248712, 194983, 194983, 194983, 194983, 194983, 194983, 232325, 232325];
    for id in ids {
        let s = tokenizer
            .decode(&[id], false)
            .map_err(|e| anyhow::anyhow!("failed to decode token {}: {}", id, e))?;
        println!("{}: {:?}", id, s);
    }
    Ok(())
}
