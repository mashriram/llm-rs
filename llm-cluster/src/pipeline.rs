use std::net::SocketAddr;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, bail};

#[derive(Debug, Serialize, Deserialize)]
pub struct ActivationMessage {
    pub seq_ids: Vec<u64>,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Upper bound on the declared length prefix for an incoming activation
/// message. Peer-controlled length fields must never be trusted directly for
/// allocation sizing -- an absurd/malicious value would otherwise trigger a
/// huge allocation attempt (OOM/abort) before we've even validated the
/// message. 1 GiB comfortably covers realistic activation tensor sizes while
/// still bounding worst-case memory blowup from a single message.
const MAX_MESSAGE_LEN_BYTES: u64 = 1 << 30;

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
        if len > MAX_MESSAGE_LEN_BYTES {
            bail!(
                "Peer declared an activation message length of {} bytes, exceeding the \
                 maximum allowed {} bytes; rejecting to avoid an unbounded allocation",
                len, MAX_MESSAGE_LEN_BYTES
            );
        }
        let mut buf = vec![0; len as usize];
        stream.read_exact(&mut buf).await?;

        let msg: ActivationMessage = serde_json::from_slice(&buf)?;

        // Cross-validate the deserialized shape against the actual data length.
        // Both come straight from an untrusted peer over the network; without
        // this check a mismatched shape/data pair would defer to a later panic
        // (e.g. in whatever tensor-construction code consumes `msg.data`
        // reshaped by `msg.shape`) instead of a clean error here.
        let expected_len: usize = msg.shape.iter().try_fold(1usize, |acc, &d| {
            acc.checked_mul(d)
        }).ok_or_else(|| anyhow::anyhow!(
            "ActivationMessage shape {:?} overflows when computing element count",
            msg.shape
        ))?;
        if expected_len != msg.data.len() {
            bail!(
                "ActivationMessage shape {:?} implies {} elements but data has {} elements",
                msg.shape, expected_len, msg.data.len()
            );
        }

        Ok(msg)
    }
}
