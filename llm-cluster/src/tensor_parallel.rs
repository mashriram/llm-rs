use anyhow::{Result, bail};
use candle_core::Tensor;

/// Compute the contiguous `[start, end)` index range owned by `rank` out of
/// `world_size` shards over a dimension of size `dim_size`, distributing any
/// remainder (`dim_size % world_size`) across the first `remainder` ranks
/// instead of dropping it. Every index in `0..dim_size` is assigned to
/// exactly one rank -- e.g. for `dim_size=10, world_size=3` this yields
/// shard sizes `[4, 3, 3]` rather than silently dropping index 9.
fn shard_bounds(dim_size: usize, rank: usize, world_size: usize) -> Result<(usize, usize)> {
    if world_size == 0 {
        bail!("world_size must be > 0 when computing tensor-parallel shard bounds");
    }
    let base = dim_size / world_size;
    let remainder = dim_size % world_size;
    let start = rank * base + rank.min(remainder);
    let extra = if rank < remainder { 1 } else { 0 };
    Ok((start, start + base + extra))
}

/// Slice a weight tensor for Column Parallelism (shards along output dimension).
pub fn shard_col_parallel(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    let out_dim = weight.dim(0)?;
    let (start, end) = shard_bounds(out_dim, rank, world_size)?;
    Ok(weight.narrow(0, start, end - start)?)
}

/// Slice a weight tensor for Row Parallelism (shards along input dimension).
pub fn shard_row_parallel(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    let in_dim = weight.dim(1)?;
    let (start, end) = shard_bounds(in_dim, rank, world_size)?;
    Ok(weight.narrow(1, start, end - start)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_bounds_covers_every_index_when_uneven() {
        // out_dim=10, world_size=3 -> sizes [4,3,3], no index dropped.
        let dim_size = 10;
        let world_size = 3;
        let mut covered = vec![false; dim_size];
        let mut total = 0usize;
        for rank in 0..world_size {
            let (start, end) = shard_bounds(dim_size, rank, world_size).unwrap();
            for i in start..end {
                assert!(!covered[i], "index {} assigned to more than one rank", i);
                covered[i] = true;
            }
            total += end - start;
        }
        assert_eq!(total, dim_size);
        assert!(covered.iter().all(|&c| c), "index 9 (or others) must not be dropped");
    }

    #[test]
    fn shard_bounds_rejects_zero_world_size() {
        assert!(shard_bounds(10, 0, 0).is_err());
    }

    #[test]
    fn shard_bounds_even_split() {
        let (start, end) = shard_bounds(12, 1, 4).unwrap();
        assert_eq!(end - start, 3);
    }
}
