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
    let mut sys = System::new();
    sys.refresh_memory();
    
    let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let available_memory_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    // 2. Estimate CPU GFLOPS via a dummy matrix multiplication benchmark
    let n = 128;
    let a = vec![1.0f32; n * n];
    let b = vec![2.0f32; n * n];
    let mut c = vec![0.0f32; n * n];

    let start = Instant::now();
    
    // Perform 10 iterations of GEMM (2 * N^3 operations per GEMM)
    let iterations = 10;
    for _ in 0..iterations {
        c.fill(0.0);
        for i in 0..n {
            for k in 0..n {
                let a_val = a[i * n + k];
                for j in 0..n {
                    c[i * n + j] += a_val * b[k * n + j];
                }
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
