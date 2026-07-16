use std::fs::File;
use candle_core::quantized::gguf_file;

fn main() -> anyhow::Result<()> {
    let path = "models/google_gemma-4-E2B-it-mmproj-BF16.gguf";
    let mut file = File::open(path)?;
    let content = gguf_file::Content::read(&mut file)?;
    println!("Metadata keys:");
    for (k, v) in content.metadata.iter() {
        println!("  {}: {:?}", k, v);
    }
    Ok(())
}
