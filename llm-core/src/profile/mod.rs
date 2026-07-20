//! Hardware profiling: detect available devices, memory budgets, and CPU capabilities.
//!
//! Provides:
//! - `HardwareProfile` — one-time detection of GPU/CPU memory and backend choice
//! - `CpuSimdCaps` — CPU SIMD feature detection for diagnostic logging
//!
//! # Usage
//! ```
//! use llm_core::profile::HardwareProfile;
//! let profile = HardwareProfile::get();
//! let model_bytes = 4_000_000_000;
//! let target = profile.choose_device(model_bytes);
//! ```

use std::process::Command;
use std::sync::OnceLock;
use tracing::{info, warn};
use sysinfo::{System, SystemExt};

// ---------------------------------------------------------------------------
// BackendChoice
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Cuda,
    Metal,
    Cpu,
}

// ---------------------------------------------------------------------------
// CpuSimdCaps — Phase 1.1
// ---------------------------------------------------------------------------

/// CPU SIMD capabilities detected at runtime.
///
/// Logged once at startup so bug reports can surface hardware context.
/// This does NOT add custom SIMD kernels — Candle uses these internally;
/// we surface them for observability only.
#[derive(Debug, Clone, Copy)]
pub struct CpuSimdCaps {
    pub avx2: bool,
    pub avx512f: bool,
    pub neon: bool,
}

impl CpuSimdCaps {
    pub fn detect() -> Self {
        Self {
            avx2: {
                #[cfg(target_arch = "x86_64")]
                { is_x86_feature_detected!("avx2") }
                #[cfg(not(target_arch = "x86_64"))]
                { false }
            },
            avx512f: {
                #[cfg(target_arch = "x86_64")]
                { is_x86_feature_detected!("avx512f") }
                #[cfg(not(target_arch = "x86_64"))]
                { false }
            },
            neon: cfg!(target_arch = "aarch64"),
        }
    }

    /// Log detected capabilities once at startup.
    pub fn log(&self) {
        info!(
            "CPU SIMD caps: AVX2={}, AVX-512F={}, NEON={}",
            self.avx2, self.avx512f, self.neon
        );
        if !self.avx2 && !self.neon {
            warn!(
                "No AVX2 or NEON detected. CPU inference will be significantly slower. \
                 Consider using a machine with AVX2 support."
            );
        }
    }
}

// ---------------------------------------------------------------------------
// HardwareProfile
// ---------------------------------------------------------------------------

/// Immutable hardware snapshot taken once at process startup.
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub gpu_vram_total_bytes: Option<u64>,
    pub gpu_vram_free_bytes: Option<u64>,
    pub system_ram_total_bytes: u64,
    pub system_ram_free_bytes: u64,
    pub cpu_cores: usize,
    pub backend: BackendChoice,
    pub simd: CpuSimdCaps,
}

static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();

impl HardwareProfile {
    /// Get or initialize the global hardware profile (detects once, then cached).
    pub fn get() -> &'static Self {
        PROFILE.get_or_init(Self::detect)
    }

    fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let system_ram_total_bytes = sys.total_memory();
        let system_ram_free_bytes = sys.free_memory();
        let cpu_cores = sys.cpus().len();
        let simd = CpuSimdCaps::detect();
        simd.log();

        let mut gpu_vram_total_bytes = None;
        let mut gpu_vram_free_bytes = None;
        let mut backend = BackendChoice::Cpu;

        if candle_core::utils::cuda_is_available() {
            // Step 1: try nvidia-smi (most reliable, gives MiB precision).
            if let Some((total_mib, free_mib)) = query_nvidia_smi() {
                gpu_vram_total_bytes = Some(total_mib * 1024 * 1024);
                gpu_vram_free_bytes = Some(free_mib * 1024 * 1024);
                backend = BackendChoice::Cuda;
                info!("GPU: {} MiB VRAM total, {} MiB free (via nvidia-smi)", total_mib, free_mib);
            } else if let Some((total_bytes, free_bytes)) = query_cuda_driver() {
                gpu_vram_total_bytes = Some(total_bytes);
                gpu_vram_free_bytes = Some(free_bytes);
                backend = BackendChoice::Cuda;
                info!(
                    "GPU: {:.2} MiB VRAM total, {:.2} MiB free (via CUDA driver API)",
                    total_bytes as f64 / 1024.0 / 1024.0,
                    free_bytes as f64 / 1024.0 / 1024.0
                );
            } else {
                // Step 3: nvidia-smi and driver API both failed (e.g. no driver/libs).
                // Refuse to guess — fall back to CPU with a clear warning.
                // Guessing 8 GB on a 4 GB card or refusing to use a 24 GB card is equally wrong.
                warn!(
                    "CUDA is available but nvidia-smi query and CUDA driver API both failed. \
                     Cannot determine free VRAM safely. Falling back to CPU. Set LLM_FORCE_CUDA=1 to override (risk: OOM)."
                );
                if std::env::var("LLM_FORCE_CUDA").is_ok() {
                    // User explicitly opts in; assume conservative 4 GB.
                    gpu_vram_total_bytes = Some(4 * 1024 * 1024 * 1024);
                    gpu_vram_free_bytes = Some(3 * 1024 * 1024 * 1024);
                    backend = BackendChoice::Cuda;
                    warn!("LLM_FORCE_CUDA set — assuming 4 GB VRAM. OOM risk if model is larger.");
                }
            }
        } else if let Some((total, free)) = query_metal_vram() {
            gpu_vram_total_bytes = Some(total);
            gpu_vram_free_bytes = Some(free);
            backend = BackendChoice::Metal;
            info!(
                "GPU: {:.2} MiB Metal Unified Memory limit, {:.2} MiB recommended free (via Metal API)",
                total as f64 / 1024.0 / 1024.0,
                free as f64 / 1024.0 / 1024.0
            );
        } else {
            info!("Running on CPU.");
        }

        Self {
            gpu_vram_total_bytes,
            gpu_vram_free_bytes,
            system_ram_total_bytes,
            system_ram_free_bytes,
            cpu_cores,
            backend,
            simd,
        }
    }

    /// Choose where to run a model of the given estimated byte size.
    ///
    /// Applies a **15% safety headroom** to account for KV-cache growth during
    /// long sequences. On CUDA, checks free VRAM. On CPU, checks free RAM and
    /// refuses to load if the model would OOM (clean error > OS kill).
    pub fn choose_device(&self, estimated_bytes: u64) -> Result<BackendChoice, String> {
        if std::env::var("LLM_FORCE_CPU").is_ok() {
            info!("LLM_FORCE_CPU is set → forcing CPU execution.");
            return Ok(BackendChoice::Cpu);
        }

        // 15% headroom for KV cache growth and fragmentation.
        let required = (estimated_bytes as f64 * 1.15) as u64;

        if self.backend == BackendChoice::Cuda || self.backend == BackendChoice::Metal {
            if let Some(free_bytes) = self.gpu_vram_free_bytes {
                if required < free_bytes {
                    info!(
                        "Model: {:.2} MB (with 15% headroom: {:.2} MB). Free VRAM/Unified Memory: {:.2} MB. → {:?}.",
                        mb(estimated_bytes), mb(required), mb(free_bytes), self.backend
                    );
                    return Ok(self.backend);
                }
                warn!(
                    "Model: {:.2} MB (with 15% headroom: {:.2} MB) exceeds VRAM/Unified Memory: {:.2} MB. → CPU.",
                    mb(estimated_bytes), mb(required), mb(free_bytes)
                );
            }
        }

        // CPU path: guard against RAM OOM.
        if required >= self.system_ram_free_bytes {
            return Err(format!(
                "Model requires {:.2} MB (with 15% headroom) but only {:.2} MB RAM is free. \
                 Aborting to prevent OOM. Free up memory or use a smaller/quantized model.",
                mb(required), mb(self.system_ram_free_bytes)
            ));
        }

        Ok(BackendChoice::Cpu)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Query nvidia-smi for (total_MiB, free_MiB). Returns None on any failure.
fn query_nvidia_smi() -> Option<(u64, u64)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut best: Option<(u64, u64)> = None;
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
        if parts.len() == 2 {
            if let (Ok(total), Ok(free)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                if best.map(|(_, best_free)| free > best_free).unwrap_or(true) {
                    best = Some((total, free));
                }
            }
        }
    }
    best
}

fn mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

#[cfg(feature = "cuda")]
fn query_cuda_driver() -> Option<(u64, u64)> {
    use candle_core::cuda_backend::cudarc::driver::result::mem_get_info;
    if let Ok((free, total)) = mem_get_info() {
        Some((total as u64, free as u64))
    } else {
        None
    }
}

#[cfg(not(feature = "cuda"))]
fn query_cuda_driver() -> Option<(u64, u64)> {
    None
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn query_metal_vram() -> Option<(u64, u64)> {
    if candle_core::Device::new_metal(0).is_ok() {
        if let Some(dev) = metal::Device::system_default() {
            let recommended = dev.recommended_max_working_set_size();
            let current_allocated = dev.current_allocated_size();
            let free = recommended.saturating_sub(current_allocated);
            Some((recommended, free))
        } else {
            None
        }
    } else {
        None
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn query_metal_vram() -> Option<(u64, u64)> {
    None
}
