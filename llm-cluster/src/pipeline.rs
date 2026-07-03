use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationMessage {
    pub seq_ids: Vec<u64>,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct PipelineStageSender {
    addr: SocketAddr,
}

impl PipelineStageSender {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    /// Send activations to the next stage in the pipeline.
    pub async fn send(&self, msg: &ActivationMessage) -> Result<()> {
        let mut stream = TcpStream::connect(self.addr).await
            .context("Failed to connect to next pipeline stage")?;
        
        let bytes = serde_json::to_vec(msg)?;
        let len = bytes.len() as u64;
        
        // Write length prefix first
        stream.write_u64(len).await?;
        stream.write_all(&bytes).await?;
        stream.flush().await?;
        Ok(())
    }
}

pub struct PipelineStageReceiver {
    listener: TcpListener,
}

impl PipelineStageReceiver {
    pub async fn bind(addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(addr).await
            .context("Failed to bind pipeline receiver port")?;
        Ok(Self { listener })
    }

    /// Receive activations from the previous stage in the pipeline.
    pub async fn recv(&self) -> Result<ActivationMessage> {
        let (mut stream, _) = self.listener.accept().await?;
        
        let len = stream.read_u64().await?;
        let mut buf = vec![0; len as usize];
        stream.read_exact(&mut buf).await?;
        
        let msg: ActivationMessage = serde_json::from_slice(&buf)?;
        Ok(msg)
    }
}
