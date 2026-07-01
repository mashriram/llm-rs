use cubecl::prelude::*;

#[cube(launch)]
pub fn rope_kernel<F: Float>(
    q: &mut Tensor<F>,
    k: &mut Tensor<F>,
    positions: &Tensor<u32>,
    rope_theta: F,
    #[comptime] head_dim: u32,
) {
    let token_idx = ABSOLUTE_POS_X;
    let num_tokens = positions.len();
    
    if token_idx < num_tokens {
        let pos = F::cast_from(positions[token_idx]);
        let head_idx = ABSOLUTE_POS_Y;
        let num_heads = q.shape(1);
        
        if head_idx < num_heads {
            for i in 0..(head_dim / 2) {
                let idx_f = F::cast_from(2 * i);
                let head_dim_f = F::cast_from(head_dim);
                let exponent = idx_f / head_dim_f;
                let theta = pos / F::powf(rope_theta, exponent);
                
                let cos_theta = F::cos(theta);
                let sin_theta = F::sin(theta);
                
                let base_idx = token_idx * (num_heads * head_dim) + head_idx * head_dim;
                let idx_1 = base_idx + 2 * i;
                let idx_2 = idx_1 + 1;
                
                let q1 = q[idx_1];
                let q2 = q[idx_2];
                q[idx_1] = q1 * cos_theta - q2 * sin_theta;
                q[idx_2] = q1 * sin_theta + q2 * cos_theta;
                
                let k1 = k[idx_1];
                let k2 = k[idx_2];
                k[idx_1] = k1 * cos_theta - k2 * sin_theta;
                k[idx_2] = k1 * sin_theta + k2 * cos_theta;
            }
        }
    }
}
