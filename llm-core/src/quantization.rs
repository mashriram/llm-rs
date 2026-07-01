/// Dequantize Q8_0 values.
pub fn dequant_q8_0(i8_values: &[i8], scale: f32) -> Vec<f32> {
    i8_values.iter().map(|&v| v as f32 * scale).collect()
}

/// Parse a Q8_0 block to extract scale and values.
pub fn parse_q8_0_block(block: &[u8]) -> (f32, Vec<i8>) {
    if block.len() < 2 {
        return (0.0, Vec::new());
    }
    // First 2 bytes are f16 scale
    let scale_bytes = [block[0], block[1]];
    let scale_f16 = half::f16::from_le_bytes(scale_bytes);
    let scale = scale_f16.to_f32();
    
    // Remaining bytes are i8 values
    let i8_values = block[2..].iter().map(|&b| b as i8).collect();
    (scale, i8_values)
}
