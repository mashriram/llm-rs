use cubecl::prelude::*;

#[cube(launch)]
pub fn quantized_gemv_kernel<F: Float>(
    weights: &Tensor<i8>,
    scales: &Tensor<F>,
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] hidden_dim: u32,
) {
    let row = ABSOLUTE_POS_X;
    let out_dim = output.shape(0);

    if row < out_dim {
        let mut acc = F::new(0.0);
        let scale = scales[row];

        for col in 0..hidden_dim {
            let w_val = F::cast_from(weights[row * hidden_dim + col]);
            let i_val = input[col];
            acc += (w_val * scale) * i_val;
        }

        output[row] = acc;
    }
}
