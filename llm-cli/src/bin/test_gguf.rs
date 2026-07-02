fn main() -> anyhow::Result<()> {
    let mut file = std::fs::File::open("/home/mukundan/learning/llm/gemma-4-E2B-it-Q4_K_M.gguf")?;
    let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    println!("Gemma4 Layer 0 PLE tensor shapes:");
    for suffix in &["inp_gate.weight", "proj.weight", "post_norm.weight", "layer_output_scale.weight"] {
        let name = format!("blk.0.{}", suffix);
        if let Some(info) = model.tensor_infos.get(&name) {
            println!("  {}: {:?}", name, info.shape);
        }
    }
    Ok(())
}
