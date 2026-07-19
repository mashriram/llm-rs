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

    // Sort nodes by ID to ensure deterministic, contiguous layer-range
    // assignment (which node gets which layer indices).
    let mut node_list: Vec<(&String, &NodeCapability)> = nodes.iter().collect();
    node_list.sort_by(|a, b| a.0.cmp(b.0));

    let total_gflops: f64 = node_list.iter().map(|(_, cap)| cap.cpu_gflops).sum();

    // First pass: give every node its proportional floor allocation. Using
    // `floor` (rather than `round`) here is deliberate -- it guarantees
    // `sum(alloc) <= n_layers`, so the leftover from rounding is always a
    // non-negative "remainder" of whole layers to hand out explicitly below,
    // rather than by whichever node happens to sort last alphabetically.
    let mut alloc: Vec<usize> = node_list
        .iter()
        .map(|(_, cap)| {
            let fraction = if total_gflops > 0.0 {
                cap.cpu_gflops / total_gflops
            } else {
                1.0 / node_list.len() as f64
            };
            ((meta.n_layers as f64) * fraction).floor() as usize
        })
        .collect();

    let allocated_sum: usize = alloc.iter().sum();
    let mut remainder = meta.n_layers.saturating_sub(allocated_sum);

    // Second pass: hand out the rounding remainder to the highest-GFLOPS
    // node(s) first (round-robin if it exceeds one layer per node), so the
    // extra layers go to whichever node is actually most capable rather than
    // to an arbitrary alphabetical tie-break.
    let mut gflops_order: Vec<usize> = (0..node_list.len()).collect();
    gflops_order.sort_by(|&a, &b| {
        node_list[b].1.cpu_gflops.total_cmp(&node_list[a].1.cpu_gflops)
    });

    let mut oi = 0;
    while remainder > 0 && !gflops_order.is_empty() {
        let idx = gflops_order[oi % gflops_order.len()];
        alloc[idx] += 1;
        remainder -= 1;
        oi += 1;
    }

    // Build the final contiguous partitions in node-ID order using the
    // (gflops-aware) allocation counts computed above.
    let mut partitions = Vec::new();
    let mut current_layer = 0;
    for (i, (node_id, _)) in node_list.iter().enumerate() {
        let end_layer = (current_layer + alloc[i]).min(meta.n_layers);
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

#[cfg(test)]
mod tests {
    use super::*;
    use llm_core::types::{HiddenAct, WeightDtype};

    fn mock_meta(n_layers: usize) -> ModelMeta {
        ModelMeta {
            vocab_size: 32000,
            hidden_dim: 4096,
            n_layers,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate_dim: 11008,
            max_seq_len: 2048,
            rope_theta: 10000.0,
            weight_dtype: WeightDtype::F16,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            hidden_act: HiddenAct::SiLU,
            no_rope_layers: vec![false; n_layers],
            has_vision_encoder: false,
            vision_hidden_dim: None,
            vision_patch_size: None,
            vision_image_size: None,
            vision_num_layers: None,
            vision_num_heads: None,
            vision_projection_dim: None,
            spatial_merge_size: None,
            is_deepstack_layers: None,
            projector_type: None,
            has_audio_encoder: false,
            audio_hidden_dim: None,
            audio_block_count: None,
            audio_embedding_length: None,
            audio_num_mel_bins: None,
            shared_kv_layers: None,
            sliding_window_pattern: None,
            sliding_window: None,
            key_length: None,
            key_length_swa: None,
            rope_theta_swa: None,
            final_logit_softcapping: None,
            is_gemma: false,
            ple_dim: None,
            embed_scale: None,
            arch: "mock".to_string(),
            chat_template: None,
            eos_token_str: None,
        }
    }

    fn cap(cpu_gflops: f64) -> NodeCapability {
        NodeCapability {
            total_memory_gb: 16.0,
            available_memory_gb: 8.0,
            cpu_gflops,
        }
    }

    #[test]
    fn remainder_goes_to_highest_gflops_node_not_alphabetically_last() {
        let meta = mock_meta(10);
        let mut nodes = HashMap::new();
        // Alphabetically "z-node" sorts last, but "a-node" is far more capable.
        // Equal-ish shares with a remainder: 10 layers / 3 nodes.
        nodes.insert("a-node".to_string(), cap(100.0));
        nodes.insert("m-node".to_string(), cap(10.0));
        nodes.insert("z-node".to_string(), cap(10.0));

        let partitions = partition_model(&meta, &nodes);

        // All layers must be assigned, contiguously, with no gaps/overlaps.
        let total: usize = partitions.iter().map(|p| p.end_layer - p.start_layer).sum();
        assert_eq!(total, 10);

        let a_layers = partitions.iter().find(|p| p.node_id == "a-node").unwrap();
        let a_count = a_layers.end_layer - a_layers.start_layer;

        let z_layers = partitions.iter().find(|p| p.node_id == "z-node");
        let z_count = z_layers.map(|p| p.end_layer - p.start_layer).unwrap_or(0);

        // The high-gflops node should get strictly more layers than the
        // alphabetically-last low-gflops node; previously the bug gave the
        // remainder to whichever node sorted last regardless of capability.
        assert!(a_count > z_count, "expected a-node ({}) > z-node ({})", a_count, z_count);
    }

    #[test]
    fn all_layers_assigned_when_evenly_divisible() {
        let meta = mock_meta(9);
        let mut nodes = HashMap::new();
        nodes.insert("n1".to_string(), cap(10.0));
        nodes.insert("n2".to_string(), cap(10.0));
        nodes.insert("n3".to_string(), cap(10.0));

        let partitions = partition_model(&meta, &nodes);
        let total: usize = partitions.iter().map(|p| p.end_layer - p.start_layer).sum();
        assert_eq!(total, 9);
    }
}
