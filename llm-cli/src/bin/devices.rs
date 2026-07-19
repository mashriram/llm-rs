//! `llm devices` — print this machine's auto-detected HardwareProfile.
//!
//! Per goal.md's CLI contract: report what HardwareProfile picked, with no
//! model loaded, so a user can confirm hardware detection before serving.

fn main() {
    tracing_subscriber::fmt::init();
    let profile = llm_core::profile::HardwareProfile::get();

    println!("llm-rs hardware profile");
    println!("=======================");
    println!("Selected backend : {:?}", profile.backend);
    println!("CPU cores        : {}", profile.cpu_cores);
    println!(
        "CPU SIMD         : AVX2={} AVX-512F={} NEON={}",
        profile.simd.avx2, profile.simd.avx512f, profile.simd.neon
    );
    println!(
        "System RAM       : {:.2} GB total, {:.2} GB free",
        profile.system_ram_total_bytes as f64 / 1e9,
        profile.system_ram_free_bytes as f64 / 1e9
    );
    match (profile.gpu_vram_total_bytes, profile.gpu_vram_free_bytes) {
        (Some(total), Some(free)) => {
            let label = match profile.backend {
                llm_core::profile::BackendChoice::Metal => "Unified Memory",
                _ => "VRAM",
            };
            println!(
                "GPU {label:16}: {:.2} GB total, {:.2} GB free",
                total as f64 / 1e9,
                free as f64 / 1e9
            );
        }
        _ => println!("GPU              : none detected"),
    }
}
