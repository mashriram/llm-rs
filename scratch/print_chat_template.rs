fn main() -> anyhow::Result<()> {
    let model_path = "/home/shri/learning/llm-rs/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf";
    let mut file = std::fs::File::open(model_path)?;
    let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    println!("=== GGUF Tensors ===");
    let mut tensors: Vec<_> = content.tensor_infos.keys().cloned().collect();
    tensors.sort();
    for t in tensors {
        println!("{}", t);
    }
    
    Ok(())
}
