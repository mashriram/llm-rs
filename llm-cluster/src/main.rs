pub mod profiler;
pub mod analyzer;
pub mod pipeline;
pub mod moe;
pub mod recovery;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::Duration;
use clap::{Parser, Subcommand};
use tokio::time::sleep;
use tracing::{info, warn};

use profiler::{profile_node, NodeCapability};
use recovery::ClusterHealthMonitor;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start this node as a coordinator
    Coordinator {
        #[arg(long, default_value = "127.0.0.1:9000")]
        listen_addr: SocketAddr,
    },
    /// Start this node as a worker and connect to a coordinator
    Worker {
        #[arg(long, default_value = "127.0.0.1:9000")]
        coordinator_addr: SocketAddr,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // NOTE (audit finding #16): `llm-cluster`'s Coordinator/Worker binaries are
    // scaffolding only. There is no real networking here yet: no
    // `TcpListener::bind`/`TcpStream::connect` between coordinator and worker,
    // `ClusterHealthMonitor::record_heartbeat` is never called from anywhere
    // (so `check_failures` can never actually observe a real heartbeat, let
    // alone its absence), and "Pause-Replicate-Retry" is currently just a log
    // line with no actual pause, replicate, or retry behavior. Building a real
    // clustering network stack is out of scope for this pass; this warning
    // exists so nobody mistakes these log lines for a working cluster.
    warn!(
        "llm-cluster is NOT yet functional: no real networking is implemented \
         (no TcpListener/TcpStream between coordinator and worker, heartbeats \
         are never actually sent/recorded, failure recovery is a log line \
         only). This binary is scaffolding only -- do not deploy expecting \
         working clustering."
    );

    match cli.command {
        Commands::Coordinator { listen_addr } => {
            info!("Starting cluster coordinator on {} (scaffolding -- not a real listener)", listen_addr);

            let mut health_monitor = ClusterHealthMonitor::new(Duration::from_secs(3));
            let mut active_nodes: HashMap<String, NodeCapability> = HashMap::new();

            // Coordinator loop: Monitor health and orchestrate work
            tokio::spawn(async move {
                loop {
                    sleep(Duration::from_millis(500)).await;
                    let failures = health_monitor.check_failures();
                    for node_id in failures {
                        active_nodes.remove(&node_id);
                        info!("Active nodes remaining: {:?}", active_nodes.keys().collect::<Vec<_>>());
                    }
                }
            });

            // Keep coordinator alive (mock server listener -- accepts no connections)
            loop {
                sleep(Duration::from_secs(3600)).await;
            }
        }
        Commands::Worker { coordinator_addr } => {
            info!("Starting cluster worker (mock -- does not actually connect to {})", coordinator_addr);

            // Profile local node capabilities
            let cap = profile_node()?;
            info!("Local node capabilities profiled: {:?}", cap);

            // Periodically send heartbeats (mock implementation -- no network I/O occurs)
            loop {
                info!("(mock) Would send heartbeat to coordinator here; no network call is made.");
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}
