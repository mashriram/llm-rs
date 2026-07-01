use cubecl::prelude::*;

#[cube(launch)]
pub fn paged_attention_kernel<F: Float>(
    q: &Tensor<F>,
    block_table: &Tensor<u32>,
    k_cache: &Tensor<F>,
    v_cache: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] block_size: u32,
    #[comptime] n_heads: u32,
    #[comptime] head_dim: u32,
) {
    let seq_idx = ABSOLUTE_POS_X;
    let num_seqs = q.shape(0);
    
    if seq_idx < num_seqs {
        let head_idx = ABSOLUTE_POS_Y;
        if head_idx < n_heads {
            let num_blocks = block_table.shape(1);
            
            let mut max_score = F::new(-99999.0);
            let mut sum_exp = F::new(0.0);
            
            for b in 0..num_blocks {
                let physical_block = block_table[seq_idx * num_blocks + b];
                
                for t in 0..block_size {
                    let mut score = F::new(0.0);
                    for d in 0..head_dim {
                        let qval_idx = seq_idx * (n_heads * head_dim) + head_idx * head_dim + d;
                        let q_val = q[qval_idx];
                        let k_idx = physical_block * (block_size * n_heads * head_dim)
                            + t * (n_heads * head_dim)
                            + head_idx * head_dim
                            + d;
                        let k_val = k_cache[k_idx];
                        score += q_val * k_val;
                    }
                    score = score / F::sqrt(F::cast_from(head_dim));
                    
                    if score > max_score {
                        let scale = F::exp(max_score - score);
                        sum_exp = sum_exp * scale + F::new(1.0);
                        max_score = score;
                    } else {
                        let scale = F::exp(score - max_score);
                        sum_exp += scale;
                    }
                }
            }
            
            for d in 0..head_dim {
                let mut out_val = F::new(0.0);
                for b in 0..num_blocks {
                    let physical_block = block_table[seq_idx * num_blocks + b];
                    
                    for t in 0..block_size {
                        let mut score = F::new(0.0);
                        for dk in 0..head_dim {
                            let qval_idx = seq_idx * (n_heads * head_dim) + head_idx * head_dim + dk;
                            let q_val = q[qval_idx];
                            let k_idx = physical_block * (block_size * n_heads * head_dim)
                                + t * (n_heads * head_dim)
                                + head_idx * head_dim
                                + dk;
                            let k_val = k_cache[k_idx];
                            score += q_val * k_val;
                        }
                        score = score / F::sqrt(F::cast_from(head_dim));
                        
                        let weight = F::exp(score - max_score) / sum_exp;
                        
                        let v_idx = physical_block * (block_size * n_heads * head_dim)
                            + t * (n_heads * head_dim)
                            + head_idx * head_dim
                            + d;
                        let v_val = v_cache[v_idx];
                        out_val += weight * v_val;
                    }
                }
                let out_idx = seq_idx * (n_heads * head_dim) + head_idx * head_dim + d;
                output[out_idx] = out_val;
            }
        }
    }
}
