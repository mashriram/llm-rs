use anyhow::{Result, bail};

/// Exact on-disk size of one GGML Q8_0 block: 2-byte f16 scale + 32 i8 values.
pub const Q8_0_BLOCK_SIZE: usize = 34;

/// Dequantize Q8_0 values.
pub fn dequant_q8_0(i8_values: &[i8], scale: f32) -> Vec<f32> {
    i8_values.iter().map(|&v| v as f32 * scale).collect()
}

/// Parse a Q8_0 block to extract scale and values.
///
/// A GGML Q8_0 block is EXACTLY 34 bytes (2-byte f16 scale + 32 i8 values) —
/// anything else is not a valid Q8_0 block. Previously this silently returned
/// `(0.0, vec![])` for any input `< 2` bytes, which would make a caller think
/// it had successfully parsed a zero-valued block instead of learning the
/// input was malformed/truncated.
pub fn parse_q8_0_block(block: &[u8]) -> Result<(f32, Vec<i8>)> {
    if block.len() != Q8_0_BLOCK_SIZE {
        bail!(
            "invalid Q8_0 block: expected exactly {} bytes, got {}",
            Q8_0_BLOCK_SIZE, block.len()
        );
    }
    // First 2 bytes are f16 scale
    let scale_bytes = [block[0], block[1]];
    let scale_f16 = half::f16::from_le_bytes(scale_bytes);
    let scale = scale_f16.to_f32();

    // Remaining 32 bytes are i8 values
    let i8_values = block[2..].iter().map(|&b| b as i8).collect();
    Ok((scale, i8_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_q8_0_block_rejects_wrong_size() {
        assert!(parse_q8_0_block(&[]).is_err());
        assert!(parse_q8_0_block(&[0u8; 2]).is_err());
        assert!(parse_q8_0_block(&[0u8; 33]).is_err());
        assert!(parse_q8_0_block(&[0u8; 35]).is_err());
    }

    #[test]
    fn parse_q8_0_block_accepts_exact_size() {
        let mut block = [0u8; Q8_0_BLOCK_SIZE];
        // scale = 1.0 in f16
        let scale_bytes = half::f16::from_f32(1.0).to_le_bytes();
        block[0] = scale_bytes[0];
        block[1] = scale_bytes[1];
        block[2] = 5; // first i8 value
        let (scale, values) = parse_q8_0_block(&block).unwrap();
        assert_eq!(scale, 1.0);
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], 5);
    }
}
