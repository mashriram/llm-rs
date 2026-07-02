use candle_core::quantized::gguf_file;

fn main() -> anyhow::Result<()> {
    let mut file = std::fs::File::open("/home/mukundan/learning/llm/gemma-4-E2B-it-Q4_K_M.gguf")?;
    let model = gguf_file::Content::read(&mut file)?;
    
    let keys = [
        "gemma4.attention.shared_kv_layers",
        "gemma4.attention.sliding_window_pattern",
        "gemma4.attention.key_length",
        "gemma4.attention.key_length_swa",
        "gemma4.rope.freq_base",
        "gemma4.rope.freq_base_swa",
        "gemma4.attention.sliding_window",
    ];

    println!("Gemma 4 Metadata:");
    for k in &keys {
        if let Some(v) = model.metadata.get(*k) {
            println!("  {}: {:?}", k, v);
        } else {
            println!("  {}: NOT FOUND", k);
        }
    }

    println!("\nTensors in model:");
    let mut tensor_keys: Vec<String> = model.tensor_infos.keys().cloned().collect();
    tensor_keys.sort();
    for tk in &tensor_keys {
        if tk.contains("layernorm") || tk.contains("norm") {
            let t = model.tensor(&mut file, tk, &candle_core::Device::Cpu)?;
            let t_f32 = t.dequantize(&candle_core::Device::Cpu)?.to_dtype(candle_core::DType::F32)?;
            let min = t_f32.flatten_all()?.min(0)?.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
            let max = t_f32.flatten_all()?.max(0)?.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?;
            let mean = t_f32.mean_all()?.to_scalar::<f32>()?;
            println!("  {} (shape={:?}): min={}, max={}, mean={}", tk, t_f32.shape(), min, max, mean);
        }
    }
    
    Ok(())
}

