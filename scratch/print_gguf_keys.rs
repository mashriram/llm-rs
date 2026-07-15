fn main() -> anyhow::Result<()> {
    let model_path = "/home/shri/learning/llm-rs/models/google_gemma-4-E2B-it-Q4_K_M.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    println!("=== GGUF Metadata Keys ===");
    for k in content.metadata.keys() {
        println!("{}", k);
    }
    
    Ok(())
}
