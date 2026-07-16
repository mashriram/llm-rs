#[cfg(feature = "cuda")]
fn test_cuda() {
    use candle_core::cuda_backend::cudarc::driver::result::mem_get_info;
    let (free, total) = mem_get_info().unwrap();
    println!("Free: {}, Total: {}", free, total);
}

fn main() {
    #[cfg(feature = "cuda")]
    test_cuda();
}
