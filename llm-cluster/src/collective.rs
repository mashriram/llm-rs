use std::net::SocketAddr;
use tokio::net::{TcpStream, TcpListener};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use anyhow::{Result, Context};

pub struct CollectiveComm {
    rank: usize,
    world_size: usize,
    peers: Vec<SocketAddr>,
}

impl CollectiveComm {
    pub fn new(rank: usize, world_size: usize, peers: Vec<SocketAddr>) -> Self {
        Self { rank, world_size, peers }
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
        let mut next_stream = TcpStream::connect(next_addr).await
            .context("Failed to connect to next peer in ring")?;

        let listener = TcpListener::bind(self.peers[self.rank]).await?;
        let (mut prev_stream, _) = listener.accept().await?;

        let chunk_size = data.len() / self.world_size;

        // Scatter-Reduce Phase
        for step in 0..(self.world_size - 1) {
            let send_chunk_idx = (self.rank + self.world_size - step) % self.world_size;
            let recv_chunk_idx = (self.rank + self.world_size - step - 1) % self.world_size;

            let send_start = send_chunk_idx * chunk_size;
            let send_end = send_start + chunk_size;
            let send_slice = &data[send_start..send_end];

            // Convert f32 slice to bytes
            let send_bytes = unsafe {
                std::slice::from_raw_parts(send_slice.as_ptr() as *const u8, chunk_size * 4)
            };

            // Send chunk to next
            next_stream.write_all(send_bytes).await?;

            // Receive chunk from prev
            let mut recv_bytes = vec![0u8; chunk_size * 4];
            prev_stream.read_exact(&mut recv_bytes).await?;

            let recv_slice = unsafe {
                std::slice::from_raw_parts(recv_bytes.as_ptr() as *const f32, chunk_size)
            };

            // Accumulate locally
            let recv_start = recv_chunk_idx * chunk_size;
            for i in 0..chunk_size {
                data[recv_start + i] += recv_slice[i];
            }
        }

        // All-Gather Phase
        for step in 0..(self.world_size - 1) {
            let send_chunk_idx = (self.rank + 1 - step + self.world_size) % self.world_size;
            let recv_chunk_idx = (self.rank - step + self.world_size) % self.world_size;

            let send_start = send_chunk_idx * chunk_size;
            let send_end = send_start + chunk_size;
            let send_slice = &data[send_start..send_end];

            let send_bytes = unsafe {
                std::slice::from_raw_parts(send_slice.as_ptr() as *const u8, chunk_size * 4)
            };

            next_stream.write_all(send_bytes).await?;

            let mut recv_bytes = vec![0u8; chunk_size * 4];
            prev_stream.read_exact(&mut recv_bytes).await?;

            let recv_slice = unsafe {
                std::slice::from_raw_parts(recv_bytes.as_ptr() as *const f32, chunk_size)
            };

            let recv_start = recv_chunk_idx * chunk_size;
            for i in 0..chunk_size {
                data[recv_start + i] = recv_slice[i];
            }
        }

        Ok(())
    }
}
