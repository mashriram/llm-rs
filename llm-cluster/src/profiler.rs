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

/// Floor applied to the measured GFLOPS estimate. If the benchmark ever
/// produces a non-finite (NaN/Infinity) or non-positive value -- e.g. from a
/// clock anomaly making `duration` ~0 -- using it directly would silently
/// propagate into `analyzer.rs`'s fraction math and zero out a node's layer
/// allocation via `NaN.round() as usize == 0`. Clamp to this safe minimum
/// instead of ever handing out a non-finite/non-positive number.
const MIN_GFLOPS: f64 = 0.001;

/// Runs the CPU-bound GEMM micro-benchmark synchronously. Kept as a plain
/// function (rather than inlined) so it can be dispatched onto a blocking
/// thread when we're running inside a Tokio runtime.
fn run_gemm_benchmark() -> f64 {
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

    if gflops.is_finite() && gflops > 0.0 {
        gflops
    } else {
        MIN_GFLOPS
    }
}

/// Profile the local node's resources (CPU GFLOPS and memory).
///
/// The GEMM micro-benchmark below is synchronous, CPU-bound work. Running it
/// directly on a Tokio worker thread would starve every other task on that
/// thread for the benchmark's duration. When called from within a Tokio
/// runtime, dispatch it via `block_in_place` so the runtime can move other
/// work off this thread first. When called outside any runtime (e.g. plain
/// `#[test]` functions with no async context), run it inline instead --
/// `block_in_place` would panic without a runtime to hand off to.
pub fn profile_node() -> Result<NodeCapability> {
    // 1. Get memory stats
    let mut sys = System::new();
    sys.refresh_memory();

    let total_memory_gb = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0;
    let available_memory_gb = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0;

    // 2. Estimate CPU GFLOPS via a dummy matrix multiplication benchmark
    let gflops = if tokio::runtime::Handle::try_current().is_ok() {
        tokio::task::block_in_place(run_gemm_benchmark)
    } else {
        run_gemm_benchmark()
    };

    Ok(NodeCapability {
        total_memory_gb,
        available_memory_gb,
        cpu_gflops: gflops,
    })
}
