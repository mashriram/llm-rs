use anyhow::Result;
use candle_core::Tensor;

/// Slice a weight tensor for Column Parallelism (shards along output dimension).
pub fn shard_col_parallel(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    let out_dim = weight.dim(0)?;
    let shard_size = out_dim / world_size;
    let start = rank * shard_size;
    let end = start + shard_size;
    Ok(weight.narrow(0, start, end - start)?)
}

/// Slice a weight tensor for Row Parallelism (shards along input dimension).
pub fn shard_row_parallel(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    let in_dim = weight.dim(1)?;
    let shard_size = in_dim / world_size;
    let start = rank * shard_size;
    let end = start + shard_size;
    Ok(weight.narrow(1, start, end - start)?)
}
