use std::collections::HashMap;
use anyhow::{Result, bail};

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
    pub fn route(&self, logits: &[f32], num_tokens: usize) -> Result<Vec<RoutedTokens>> {
        // `logits` is network/model-derived data; validate its length before
        // ever indexing into it with `t * num_experts` below. A mismatched
        // buffer (e.g. from a malformed/truncated remote message) must
        // produce a clean error here rather than an out-of-bounds slice
        // panic that would take down the whole process.
        let required_len = num_tokens
            .checked_mul(self.num_experts)
            .ok_or_else(|| anyhow::anyhow!(
                "num_tokens ({}) * num_experts ({}) overflows usize",
                num_tokens, self.num_experts
            ))?;
        if logits.len() < required_len {
            bail!(
                "MoE router logits buffer too short: expected at least {} elements \
                 (num_tokens={} * num_experts={}) but got {}",
                required_len, num_tokens, self.num_experts, logits.len()
            );
        }

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

            // Find top-k experts. Use `total_cmp` instead of `partial_cmp().unwrap()`:
            // if a router logit is ever NaN (e.g. from an unstable/misbehaving model),
            // `partial_cmp` returns `None` and would panic here; `total_cmp` gives a
            // well-defined total order for all f32 values, including NaN, so routing
            // stays deterministic instead of crashing the whole batch.
            let mut indexed_gates: Vec<(usize, f32)> = gates.into_iter().enumerate().collect();
            indexed_gates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            for k in 0..self.top_k.min(self.num_experts) {
                let (expert_id, weight) = indexed_gates[k];
                if weight > 0.0 {
                    let entry = routing_map.entry(expert_id).or_insert_with(|| (Vec::new(), Vec::new()));
                    entry.0.push(t);
                    entry.1.push(weight);
                }
            }
        }

        Ok(routing_map.into_iter().map(|(expert_id, (token_indices, weights))| {
            RoutedTokens {
                expert_id,
                token_indices,
                weights,
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_rejects_undersized_logits_buffer_instead_of_panicking() {
        let router = ExpertRouter::new(4, 2);
        // Claims 3 tokens x 4 experts = 12 elements, but only provides 8.
        let logits = vec![0.1f32; 8];
        let result = router.route(&logits, 3);
        assert!(result.is_err());
    }

    #[test]
    fn route_succeeds_with_correctly_sized_logits_buffer() {
        let router = ExpertRouter::new(2, 1);
        let logits = vec![1.0f32, 0.0, 0.0, 1.0]; // 2 tokens x 2 experts
        let result = router.route(&logits, 2);
        assert!(result.is_ok());
    }
}
