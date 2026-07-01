use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use llm_core::types::ModelMeta;
use crate::profiler::NodeCapability;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPartition {
    pub node_id: String,
    pub start_layer: usize,
    pub end_layer: usize, // exclusive
}

/// Analyze the model and partition its layers across cluster nodes based on their GFLOPS.
pub fn partition_model(
    meta: &ModelMeta,
    nodes: &HashMap<String, NodeCapability>,
) -> Vec<ClusterPartition> {
    if nodes.is_empty() {
        return vec![ClusterPartition {
            node_id: "local".to_string(),
            start_layer: 0,
            end_layer: meta.n_layers,
        }];
    }

    // Sort nodes by ID to ensure deterministic assignment
    let mut node_list: Vec<(&String, &NodeCapability)> = nodes.iter().collect();
    node_list.sort_by(|a, b| a.0.cmp(b.0));

    let total_gflops: f64 = node_list.iter().map(|(_, cap)| cap.cpu_gflops).sum();
    let mut partitions = Vec::new();
    
    let mut current_layer = 0;
    for (i, (node_id, cap)) in node_list.iter().enumerate() {
        let fraction = if total_gflops > 0.0 { cap.cpu_gflops / total_gflops } else { 1.0 / nodes.len() as f64 };
        
        let mut allocated_layers = (meta.n_layers as f64 * fraction).round() as usize;
        
        // Ensure all layers are allocated on the last node
        if i == node_list.len() - 1 {
            allocated_layers = meta.n_layers - current_layer;
        }

        let end_layer = (current_layer + allocated_layers).min(meta.n_layers);
        if current_layer < end_layer {
            partitions.push(ClusterPartition {
                node_id: (*node_id).clone(),
                start_layer: current_layer,
                end_layer,
            });
            current_layer = end_layer;
        }
    }

    partitions
}
