use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::{TcpStream, TcpListener};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::timeout;
use anyhow::{Result, Context, bail};

/// Timeout applied to every blocking ring network operation (connect, accept,
/// read_exact). Without this, one dead/slow peer would hang the whole
/// cluster indefinitely. Chosen to be generous enough for real network
/// conditions while still failing fast on a genuinely unreachable peer.
const NET_TIMEOUT: Duration = Duration::from_secs(10);

/// Compute the contiguous `[start, end)` byte-index-agnostic element range
/// owned by chunk `idx` out of `world_size` chunks over a buffer of `len`
/// elements, distributing any remainder (`len % world_size`) across the
/// first `remainder` chunks instead of dropping it. Every index in
/// `0..len` is covered by exactly one chunk's range. Mirrors the
/// last-node-absorbs-remainder spirit of `analyzer.rs::partition_model`,
/// but spreads the extra elements across the low-indexed chunks so no
/// single rank is disproportionately overloaded.
fn chunk_bounds(len: usize, world_size: usize, idx: usize) -> (usize, usize) {
    let base = len / world_size;
    let remainder = len % world_size;
    let start = idx * base + idx.min(remainder);
    let extra = if idx < remainder { 1 } else { 0 };
    (start, start + base + extra)
}

pub struct CollectiveComm {
    rank: usize,
    world_size: usize,
    peers: Vec<SocketAddr>,
}

impl CollectiveComm {
    pub fn new(rank: usize, world_size: usize, peers: Vec<SocketAddr>) -> Result<Self> {
        if peers.len() != world_size {
            bail!(
                "CollectiveComm: peers.len() ({}) must equal world_size ({})",
                peers.len(), world_size
            );
        }
        Ok(Self { rank, world_size, peers })
    }

    /// Perform a Ring All-Reduce on a buffer of floats.
    pub async fn all_reduce(&self, data: &mut [f32]) -> Result<()> {
        if self.world_size <= 1 {
            return Ok(());
        }

        // Ring-based All-Reduce communication loop
        let next_rank = (self.rank + 1) % self.world_size;
        let prev_rank = (self.rank + self.world_size - 1) % self.world_size;

        let next_addr = self.peers[next_rank];
        let _prev_addr = self.peers[prev_rank];

        // Connect to next and accept from prev
        let mut next_stream = timeout(NET_TIMEOUT, TcpStream::connect(next_addr))
            .await
            .context("Timed out connecting to next peer in ring")?
            .context("Failed to connect to next peer in ring")?;

        let listener = TcpListener::bind(self.peers[self.rank]).await?;
        let (mut prev_stream, _) = timeout(NET_TIMEOUT, listener.accept())
            .await
            .context("Timed out waiting for previous peer to connect in ring")??;

        // Scatter-Reduce Phase
        for step in 0..(self.world_size - 1) {
            let send_chunk_idx = (self.rank + self.world_size - step) % self.world_size;
            let recv_chunk_idx = (self.rank + self.world_size - step - 1) % self.world_size;

            let (send_start, send_end) = chunk_bounds(data.len(), self.world_size, send_chunk_idx);
            let send_slice = &data[send_start..send_end];

            // Convert f32 slice to bytes without relying on raw-pointer alignment
            // casts (a `Vec<u8>` buffer is not guaranteed to be 4-byte aligned, so
            // reinterpreting it as `&[f32]` via a pointer cast would be unsound).
            let mut send_bytes = Vec::with_capacity(send_slice.len() * 4);
            for v in send_slice {
                send_bytes.extend_from_slice(&v.to_le_bytes());
            }

            // Send chunk to next
            timeout(NET_TIMEOUT, next_stream.write_all(&send_bytes))
                .await
                .context("Timed out sending chunk to next peer during scatter-reduce")??;

            // Receive chunk from prev
            let (recv_start, recv_end) = chunk_bounds(data.len(), self.world_size, recv_chunk_idx);
            let recv_len = recv_end - recv_start;
            let mut recv_bytes = vec![0u8; recv_len * 4];
            timeout(NET_TIMEOUT, prev_stream.read_exact(&mut recv_bytes))
                .await
                .context("Timed out receiving chunk from previous peer during scatter-reduce")??;

            // Accumulate locally
            for i in 0..recv_len {
                let bytes: [u8; 4] = recv_bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .context("Failed to read f32 chunk from peer during all-reduce scatter phase")?;
                data[recv_start + i] += f32::from_le_bytes(bytes);
            }
        }

        // All-Gather Phase
        for step in 0..(self.world_size - 1) {
            let send_chunk_idx = (self.rank + 1 - step + self.world_size) % self.world_size;
            let recv_chunk_idx = (self.rank - step + self.world_size) % self.world_size;

            let (send_start, send_end) = chunk_bounds(data.len(), self.world_size, send_chunk_idx);
            let send_slice = &data[send_start..send_end];

            let mut send_bytes = Vec::with_capacity(send_slice.len() * 4);
            for v in send_slice {
                send_bytes.extend_from_slice(&v.to_le_bytes());
            }

            timeout(NET_TIMEOUT, next_stream.write_all(&send_bytes))
                .await
                .context("Timed out sending chunk to next peer during all-gather")??;

            let (recv_start, recv_end) = chunk_bounds(data.len(), self.world_size, recv_chunk_idx);
            let recv_len = recv_end - recv_start;
            let mut recv_bytes = vec![0u8; recv_len * 4];
            timeout(NET_TIMEOUT, prev_stream.read_exact(&mut recv_bytes))
                .await
                .context("Timed out receiving chunk from previous peer during all-gather")??;

            for i in 0..recv_len {
                let bytes: [u8; 4] = recv_bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .context("Failed to read f32 chunk from peer during all-reduce gather phase")?;
                data[recv_start + i] = f32::from_le_bytes(bytes);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bounds_covers_every_index_with_no_gaps_or_overlaps_when_uneven() {
        // 10 elements over 3 ranks: base=3, remainder=1 -> sizes [4,3,3]
        let len = 10;
        let world_size = 3;
        let mut covered = vec![false; len];
        let mut total = 0usize;
        for idx in 0..world_size {
            let (start, end) = chunk_bounds(len, world_size, idx);
            assert!(start <= end);
            for i in start..end {
                assert!(!covered[i], "index {} covered by more than one chunk", i);
                covered[i] = true;
            }
            total += end - start;
        }
        assert_eq!(total, len, "chunk_bounds must cover every element exactly once");
        assert!(covered.iter().all(|&c| c), "no index may be left uncovered");

        // Sanity-check the actual sizes for this concrete case.
        assert_eq!(chunk_bounds(len, world_size, 0), (0, 4));
        assert_eq!(chunk_bounds(len, world_size, 1), (4, 7));
        assert_eq!(chunk_bounds(len, world_size, 2), (7, 10));
    }

    #[test]
    fn chunk_bounds_evenly_divisible() {
        let len = 12;
        let world_size = 4;
        for idx in 0..world_size {
            let (start, end) = chunk_bounds(len, world_size, idx);
            assert_eq!(end - start, 3);
        }
    }

    #[test]
    fn collective_comm_new_rejects_peer_world_size_mismatch() {
        let peers = vec!["127.0.0.1:9000".parse().unwrap(), "127.0.0.1:9001".parse().unwrap()];
        let result = CollectiveComm::new(0, 3, peers);
        assert!(result.is_err());
    }

    #[test]
    fn collective_comm_new_accepts_matching_peer_world_size() {
        let peers = vec!["127.0.0.1:9000".parse().unwrap(), "127.0.0.1:9001".parse().unwrap()];
        let result = CollectiveComm::new(0, 2, peers);
        assert!(result.is_ok());
    }
}
