use std::path::Path;
use candle_core::{Tensor, Device, DType};
use llm_core::backends::audio::AudioEncoder;

fn main() -> anyhow::Result<()> {
    println!("Loading AudioEncoder...");
    let device = Device::Cpu;
    let mmproj_path = Path::new("models/google_gemma-4-E2B-it-mmproj-BF16.gguf");
    let encoder = AudioEncoder::load(mmproj_path, &device)?;
    println!(
        "Loaded AudioEncoder: hidden_dim={}, num_layers={}, num_heads={}, projection_dim={}",
        encoder.hidden_dim, encoder.num_layers, encoder.num_heads, encoder.projection_dim
    );

    // Create a dummy audio input: (1, 128, 3000)
    let dummy_input = Tensor::zeros((1, 128, 3000), DType::F32, &device)?;
    println!("Encoding dummy audio...");
    let out = encoder.encode(&dummy_input, 3000)?;
    println!("Successfully encoded audio! Output shape: {:?}", out.shape());
    Ok(())
}
