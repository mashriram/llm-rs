use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::warn;

pub struct ClusterHealthMonitor {
    heartbeats: HashMap<String, Instant>,
    timeout: Duration,
}

impl ClusterHealthMonitor {
    pub fn new(timeout: Duration) -> Self {
        Self {
            heartbeats: HashMap::new(),
            timeout,
        }
    }

    /// Record a heartbeat from a node.
    pub fn record_heartbeat(&mut self, node_id: &str) {
        self.heartbeats.insert(node_id.to_string(), Instant::now());
    }

    /// Check for failed nodes in the cluster.
    /// Returns a list of failed node IDs.
    pub fn check_failures(&mut self) -> Vec<String> {
        let now = Instant::now();
        let mut failed_nodes = Vec::new();

        for (node_id, last_seen) in &self.heartbeats {
            if now.duration_since(*last_seen) > self.timeout {
                failed_nodes.push(node_id.clone());
            }
        }

        for node_id in &failed_nodes {
            self.heartbeats.remove(node_id);
            warn!("Node {} has failed (heartbeat timeout). Triggering Pause-Replicate-Retry.", node_id);
        }

        failed_nodes
    }
}
