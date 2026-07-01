use std::process::Command;
use std::sync::OnceLock;
use tracing::{info, warn};
use sysinfo::{System, SystemExt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Cuda,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub gpu_vram_total_bytes: Option<u64>,
    pub gpu_vram_free_bytes: Option<u64>,
    pub system_ram_total_bytes: u64,
    pub system_ram_free_bytes: u64,
    pub cpu_cores: usize,
    pub backend: BackendChoice,
}

static PROFILE: OnceLock<HardwareProfile> = OnceLock::new();

impl HardwareProfile {
    pub fn get() -> &'static Self {
        PROFILE.get_or_init(Self::detect)
    }

    fn detect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let system_ram_total_bytes = sys.total_memory();
        let system_ram_free_bytes = sys.free_memory();
        let cpu_cores = sys.cpus().len();

        let mut gpu_vram_total_bytes = None;
        let mut gpu_vram_free_bytes = None;
        let mut backend = BackendChoice::Cpu;

        if candle_core::utils::cuda_is_available() {
            // Try querying nvidia-smi
            if let Ok(output) = Command::new("nvidia-smi")
                .args(["--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"])
                .output()
            {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = stdout.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() == 2 {
                        if let (Ok(total_mib), Ok(free_mib)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                            gpu_vram_total_bytes = Some(total_mib * 1024 * 1024);
                            gpu_vram_free_bytes = Some(free_mib * 1024 * 1024);
                            backend = BackendChoice::Cuda;
                            info!("Detected CUDA GPU with {} MiB VRAM ({} MiB free)", total_mib, free_mib);
                        }
                    }
                }
            }

            if backend == BackendChoice::Cpu {
                // CUDA is available but nvidia-smi failed; assume default safe limits
                gpu_vram_total_bytes = Some(8 * 1024 * 1024 * 1024); // Assume 8GB
                gpu_vram_free_bytes = Some(6 * 1024 * 1024 * 1024);  // Assume 6GB free
                backend = BackendChoice::Cuda;
                warn!("CUDA available but nvidia-smi query failed. Assuming default 8GB VRAM.");
            }
        } else {
            info!("CUDA not detected. Running on CPU.");
        }

        Self {
            gpu_vram_total_bytes,
            gpu_vram_free_bytes,
            system_ram_total_bytes,
            system_ram_free_bytes,
            cpu_cores,
            backend,
        }
    }

    /// Dynamically choose where a model with a given size footprint should run.
    /// Incorporates a 15% safety headroom.
    pub fn choose_device(&self, estimated_bytes: u64) -> BackendChoice {
        if self.backend == BackendChoice::Cuda {
            if let Some(free_bytes) = self.gpu_vram_free_bytes {
                let required = (estimated_bytes as f64 * 1.01) as u64;
                if required < free_bytes {
                    info!(
                        "Model estimated size: {:.2} MB (required with safety margin: {:.2} MB). Free VRAM: {:.2} MB. Selecting CUDA.",
                        estimated_bytes as f64 / 1024.0 / 1024.0,
                        required as f64 / 1024.0 / 1024.0,
                        free_bytes as f64 / 1024.0 / 1024.0
                    );
                    return BackendChoice::Cuda;
                } else {
                    warn!(
                        "Model estimated size: {:.2} MB (required with safety margin: {:.2} MB) exceeds free VRAM: {:.2} MB. Falling back to CPU.",
                        estimated_bytes as f64 / 1024.0 / 1024.0,
                        required as f64 / 1024.0 / 1024.0,
                        free_bytes as f64 / 1024.0 / 1024.0
                    );
                }
            }
        }
        BackendChoice::Cpu
    }
}
