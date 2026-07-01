use std::collections::HashMap;

/// Simple Top-1/Top-2 Expert Router.
/// Matches MLC-LLM MoE routing contracts.
pub struct ExpertRouter {
    num_experts: usize,
    top_k: usize,
}

#[derive(Debug, Clone)]
pub struct RoutedTokens {
    pub expert_id: usize,
    pub token_indices: Vec<usize>,
    pub weights: Vec<f32>,
}

impl ExpertRouter {
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self { num_experts, top_k }
    }

    /// Routes a batch of tokens to the appropriate experts based on router logits.
    /// Shape of logits: [num_tokens, num_experts]
    pub fn route(&self, logits: &[f32], num_tokens: usize) -> Vec<RoutedTokens> {
        let mut routing_map: HashMap<usize, (Vec<usize>, Vec<f32>)> = HashMap::new();
        
        for t in 0..num_tokens {
            let start = t * self.num_experts;
            let end = start + self.num_experts;
            let token_logits = &logits[start..end];

            // Compute softmax to get gating weights
            let max_val = token_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_logits: Vec<f32> = token_logits.iter().map(|&l| (l - max_val).exp()).collect();
            let sum_exp: f32 = exp_logits.iter().sum();
            let gates: Vec<f32> = exp_logits.iter().map(|&e| e / sum_exp).collect();

            // Find top-k experts
            let mut indexed_gates: Vec<(usize, f32)> = gates.into_iter().enumerate().collect();
            indexed_gates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for k in 0..self.top_k.min(self.num_experts) {
                let (expert_id, weight) = indexed_gates[k];
                if weight > 0.0 {
                    let entry = routing_map.entry(expert_id).or_insert_with(|| (Vec::new(), Vec::new()));
                    entry.0.push(t);
                    entry.1.push(weight);
                }
            }
        }

        routing_map.into_iter().map(|(expert_id, (token_indices, weights))| {
            RoutedTokens {
                expert_id,
                token_indices,
                weights,
            }
        }).collect()
    }
}
