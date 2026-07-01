use cubecl::prelude::*;

#[cube(launch)]
pub fn silu_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] len: u32,
) {
    let idx = ABSOLUTE_POS_X;
    if idx < len {
        let val = input[idx];
        let sigmoid = F::recip(F::new(1.0) + F::exp(-val));
        output[idx] = val * sigmoid;
    }
}
