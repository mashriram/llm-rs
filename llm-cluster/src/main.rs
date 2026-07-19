pub mod profiler;
pub mod analyzer;
pub mod pipeline;
pub mod moe;
pub mod recovery;
pub mod protocol;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use anyhow::Context;
use clap::{Parser, Subcommand};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::time::sleep;
use tracing::{error, info, warn};

use profiler::{profile_node, NodeCapability};
use protocol::{read_message, write_message, ClusterMessage};
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
        /// Seconds without a heartbeat before a node is considered failed.
        #[arg(long, default_value_t = 5)]
        heartbeat_timeout_secs: u64,
    },
    /// Start this node as a worker and connect to a coordinator
    Worker {
        #[arg(long, default_value = "127.0.0.1:9000")]
        coordinator_addr: SocketAddr,
        /// This node's identifier. Defaults to `hostname:pid` so multiple
        /// workers on the same machine (as used in this session's own
        /// localhost testing) don't collide.
        #[arg(long)]
        node_id: Option<String>,
        /// Seconds between heartbeats sent to the coordinator.
        #[arg(long, default_value_t = 1)]
        heartbeat_interval_secs: u64,
    },
}

type ActiveNodes = Arc<Mutex<HashMap<String, NodeCapability>>>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Coordinator { listen_addr, heartbeat_timeout_secs } => {
            run_coordinator(listen_addr, heartbeat_timeout_secs).await
        }
        Commands::Worker { coordinator_addr, node_id, heartbeat_interval_secs } => {
            let node_id = node_id.unwrap_or_else(default_node_id);
            run_worker(coordinator_addr, node_id, heartbeat_interval_secs).await
        }
    }
}

fn default_node_id() -> String {
    format!("{}:{}", hostname_best_effort(), std::process::id())
}

fn hostname_best_effort() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| {
            // `hostname()` isn't in std; shell out rather than adding a
            // dependency just for this cosmetic default.
            std::process::Command::new("hostname")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown-host".to_string())
        })
}

async fn run_coordinator(listen_addr: SocketAddr, heartbeat_timeout_secs: u64) -> anyhow::Result<()> {
    let listener = TcpListener::bind(listen_addr)
        .await
        .with_context(|| format!("failed to bind coordinator listener on {listen_addr}"))?;
    info!("Cluster coordinator listening on {listen_addr}");

    let health_monitor = Arc::new(Mutex::new(ClusterHealthMonitor::new(Duration::from_secs(
        heartbeat_timeout_secs,
    ))));
    let active_nodes: ActiveNodes = Arc::new(Mutex::new(HashMap::new()));

    // Background loop: periodically check for nodes that stopped sending
    // heartbeats and evict them from `active_nodes`. This is the real
    // failure-detection path `ClusterHealthMonitor` was built for - it now
    // actually observes real network heartbeats (via the per-connection
    // handlers below) instead of never being called at all.
    {
        let health_monitor = health_monitor.clone();
        let active_nodes = active_nodes.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_millis(500)).await;
                let failures = health_monitor.lock().await.check_failures();
                if !failures.is_empty() {
                    let mut nodes = active_nodes.lock().await;
                    for node_id in &failures {
                        nodes.remove(node_id);
                    }
                    warn!(
                        "Node(s) {:?} failed (missed heartbeat timeout). Active nodes remaining: {:?}. \
                         NOTE: eviction from the active-node list is the extent of failure handling \
                         implemented so far - re-partitioning layers onto survivors and re-prefilling \
                         in-flight sequences (goal.md's full Pause-Replicate-Retry) is not yet wired up.",
                        failures,
                        nodes.keys().collect::<Vec<_>>()
                    );
                }
            }
        });
    }

    loop {
        let (stream, peer_addr) = listener.accept().await.context("accept() failed")?;
        info!("Accepted connection from {peer_addr}");
        let health_monitor = health_monitor.clone();
        let active_nodes = active_nodes.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_worker_connection(stream, health_monitor, active_nodes.clone()).await {
                warn!("Connection from {peer_addr} ended: {e:#}");
            }
        });
    }
}

async fn handle_worker_connection(
    mut stream: TcpStream,
    health_monitor: Arc<Mutex<ClusterHealthMonitor>>,
    active_nodes: ActiveNodes,
) -> anyhow::Result<()> {
    // First message on a new connection must be Hello.
    let node_id = match read_message(&mut stream).await? {
        Some(ClusterMessage::Hello { node_id, capability }) => {
            info!("Node '{node_id}' registered: {capability:?}");
            active_nodes.lock().await.insert(node_id.clone(), capability);
            health_monitor.lock().await.record_heartbeat(&node_id);
            write_message(&mut stream, &ClusterMessage::Welcome).await?;
            node_id
        }
        Some(other) => {
            anyhow::bail!("expected Hello as the first message, got {other:?}");
        }
        None => {
            anyhow::bail!("connection closed before sending Hello");
        }
    };

    let result: anyhow::Result<()> = async {
        loop {
            match read_message(&mut stream).await? {
                Some(ClusterMessage::Heartbeat { node_id: hb_id }) => {
                    if hb_id != node_id {
                        warn!(
                            "connection registered as '{node_id}' sent a heartbeat for a \
                             different node_id '{hb_id}' - ignoring the mismatched id, \
                             recording the heartbeat under the connection's own identity"
                        );
                    }
                    health_monitor.lock().await.record_heartbeat(&node_id);
                }
                Some(ClusterMessage::Hello { .. }) => {
                    warn!("node '{node_id}' sent a second Hello on an already-registered connection - ignoring");
                }
                Some(ClusterMessage::Welcome) => {
                    warn!("node '{node_id}' sent a Welcome (coordinator-only message) - ignoring");
                }
                None => {
                    info!("node '{node_id}' disconnected");
                    return Ok(());
                }
            }
        }
    }
    .await;

    // Whether the loop ended cleanly (disconnect) or with an error (bad
    // frame, I/O error), remove the node immediately rather than waiting
    // for the heartbeat-timeout sweep to notice - a closed socket is a
    // stronger, more immediate signal than a missed heartbeat.
    active_nodes.lock().await.remove(&node_id);
    result
}

async fn run_worker(coordinator_addr: SocketAddr, node_id: String, heartbeat_interval_secs: u64) -> anyhow::Result<()> {
    info!("Node '{node_id}' profiling local capabilities...");
    let capability = profile_node()?;
    info!("Local node capabilities: {capability:?}");

    // Reconnect with a fixed backoff if the coordinator is unreachable or the
    // connection drops - a real (if simple) resilience behavior, rather than
    // exiting the process the first time the coordinator is briefly down.
    loop {
        match connect_and_serve(coordinator_addr, &node_id, &capability, heartbeat_interval_secs).await {
            Ok(()) => {
                info!("Disconnected from coordinator cleanly; reconnecting in 2s...");
            }
            Err(e) => {
                error!("Connection to coordinator {coordinator_addr} failed: {e:#}; retrying in 2s...");
            }
        }
        sleep(Duration::from_secs(2)).await;
    }
}

async fn connect_and_serve(
    coordinator_addr: SocketAddr,
    node_id: &str,
    capability: &NodeCapability,
    heartbeat_interval_secs: u64,
) -> anyhow::Result<()> {
    let mut stream = TcpStream::connect(coordinator_addr)
        .await
        .with_context(|| format!("failed to connect to coordinator at {coordinator_addr}"))?;
    info!("Connected to coordinator at {coordinator_addr}");

    write_message(
        &mut stream,
        &ClusterMessage::Hello {
            node_id: node_id.to_string(),
            capability: capability.clone(),
        },
    )
    .await?;

    match read_message(&mut stream).await? {
        Some(ClusterMessage::Welcome) => info!("Registered with coordinator as '{node_id}'"),
        Some(other) => anyhow::bail!("expected Welcome from coordinator, got {other:?}"),
        None => anyhow::bail!("coordinator closed the connection before replying to Hello"),
    }

    loop {
        sleep(Duration::from_secs(heartbeat_interval_secs)).await;
        write_message(&mut stream, &ClusterMessage::Heartbeat { node_id: node_id.to_string() }).await?;
    }
}
