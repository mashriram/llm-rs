fn main() -> anyhow::Result<()> {
    println!("CUDA is available: {}", candle_core::utils::cuda_is_available());
    println!("CUDA device count: {}", candle_core::cuda_backend::device_count().unwrap_or(0));
    for i in 0..10 {
        match candle_core::Device::new_cuda(i) {
            Ok(dev) => println!("Device {}: {:?}", i, dev),
            Err(e) => println!("Device {}: Error: {:?}", i, e),
        }
    }
    Ok(())
}
