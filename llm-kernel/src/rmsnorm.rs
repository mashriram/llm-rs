use cubecl::prelude::*;

#[cube(launch)]
pub fn rms_norm_kernel<F: Float>(
    input: &Tensor<F>,
    weight: &Tensor<F>,
    output: &mut Tensor<F>,
    eps: F,
    #[comptime] rows: u32,
    #[comptime] cols: u32,
) {
    let row_idx = ABSOLUTE_POS_X;
    
    if row_idx < rows {
        let mut sum_sq = F::new(0.0);
        
        for col in 0..cols {
            let val = input[row_idx * cols + col];
            sum_sq += val * val;
        }

        let variance = sum_sq / F::cast_from(cols);
        let inv_std = F::recip(F::sqrt(variance + eps));

        for col in 0..cols {
            let val = input[row_idx * cols + col];
            let w = weight[col];
            output[row_idx * cols + col] = val * inv_std * w;
        }
    }
}
