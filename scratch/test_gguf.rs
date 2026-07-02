fn main() -> anyhow::Result<()> {
    let mut file = std::fs::File::open("/home/mukundan/learning/llm/gemma-4-E2B-it-Q4_K_M.gguf")?;
    let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    let mut keys: Vec<_> = model.metadata.keys().collect();
    keys.sort();
    
    println!("Metadata keys:");
    for k in keys {
        println!("  {}: {:?}", k, model.metadata[k]);
    }
    Ok(())
}
