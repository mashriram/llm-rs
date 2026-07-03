use candle_core::{Device, DType, Tensor, Module};
use candle_core::quantized::{QMatMul};

fn main() -> anyhow::Result<()> {
    let dev = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    println!("Selected device: {:?}", dev);
    
    let path = std::path::Path::new("/home/mukundan/learning/llm/SmolLM3-Q4_K_M.gguf");
    let mut file = std::fs::File::open(path)?;
    let model = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    
    // Find a weight tensor (e.g. blk.0.attn_q.weight)
    let weight_name = "blk.0.attn_q.weight";
    let qt = model.tensor(&mut file, weight_name, &dev)?;
    let qmatmul = QMatMul::from_qtensor(qt)?;
    
    // Test F16 input
    let x_f16 = Tensor::zeros((1, 2048), DType::F16, &dev)?;
    match qmatmul.forward(&x_f16) {
        Ok(out) => println!("F16 input works! output dtype: {:?}, shape: {:?}", out.dtype(), out.shape()),
        Err(e) => println!("F16 input failed: {:?}", e),
    }

    // Test F32 input
    let x_f32 = Tensor::zeros((1, 2048), DType::F32, &dev)?;
    match qmatmul.forward(&x_f32) {
        Ok(out) => println!("F32 input works! output dtype: {:?}, shape: {:?}", out.dtype(), out.shape()),
        Err(e) => println!("F32 input failed: {:?}", e),
    }

    Ok(())
}
