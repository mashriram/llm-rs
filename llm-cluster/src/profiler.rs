use std::time::Instant;
use serde::{Serialize, Deserialize};
use anyhow::Result;
use sysinfo::{System, SystemExt};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapability {
    pub total_memory_gb: f64,
    pub available_memory_gb: f64,
    pub cpu_gflops: f64,
}

/// Profile the local node's resources (CPU GFLOPS and memory).
pub fn profile_node() -> Result<NodeCapability> {
    // 1. Get memory stats
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let available_memory_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    // 2. Estimate CPU GFLOPS via a dummy matrix multiplication benchmark
    let n = 256;
    let mut a = vec![1.0f32; n * n];
    let mut b = vec![2.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let start = Instant::now();
    
    // Perform 50 iterations of GEMM (2 * N^3 operations per GEMM)
    let iterations = 50;
    for _ in 0..iterations {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    let duration = start.elapsed().as_secs_f64();
    
    // Total floating point operations: iterations * 2 * N^3
    let total_ops = iterations as f64 * 2.0 * (n as f64).powi(3);
    let gflops = (total_ops / duration) / 1e9;

    Ok(NodeCapability {
        total_memory_gb,
        available_memory_gb,
        cpu_gflops: gflops,
    })
}
