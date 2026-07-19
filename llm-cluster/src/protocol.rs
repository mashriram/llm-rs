//! Real wire protocol for coordinator<->worker communication.
//!
//! A simple length-prefixed JSON framing (u32 big-endian byte count + JSON
//! payload) over a plain TCP stream. This is the actual networking that was
//! missing entirely from `main.rs` before - goal.md's "USB cluster mesh" is,
//! at the transport level, just TCP/IP over whatever network interface the
//! OS exposes (a USB cable running RNDIS/CDC-ECM presents as a normal network
//! interface with its own IP), so this same protocol is what would run over
//! a real USB link - only the interface a socket binds/connects to changes,
//! not this code. See the module-level docs in `main.rs` for what has and
//! hasn't been verified (this was tested over localhost TCP in this session;
//! no physical USB link or second machine was available to test over).

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::profiler::NodeCapability;

/// Cap on a single framed message's declared size, to reject a
/// peer-controlled length prefix before allocating - the same class of
/// unbounded-allocation issue found and fixed elsewhere in this crate's
/// `pipeline.rs` for activation-tensor messages.
const MAX_MESSAGE_BYTES: u32 = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusterMessage {
    /// First message a worker sends after connecting: its identity and
    /// profiled capability, so the coordinator can register it for both
    /// health tracking and (in a future pass) layer-assignment planning.
    Hello { node_id: String, capability: NodeCapability },
    /// Periodic keep-alive; the coordinator's `ClusterHealthMonitor` treats
    /// receipt of either `Hello` or `Heartbeat` as "this node is alive".
    Heartbeat { node_id: String },
    /// Coordinator's reply to `Hello`, confirming registration.
    Welcome,
}

pub async fn write_message(stream: &mut TcpStream, msg: &ClusterMessage) -> Result<()> {
    let payload = serde_json::to_vec(msg).context("failed to serialize cluster message")?;
    let len = payload.len() as u32;
    stream.write_all(&len.to_be_bytes()).await.context("failed to write message length")?;
    stream.write_all(&payload).await.context("failed to write message body")?;
    stream.flush().await.context("failed to flush cluster stream")?;
    Ok(())
}

/// Read one length-prefixed message. Returns `Ok(None)` on a clean EOF
/// (peer closed the connection between messages, not mid-message), which
/// callers should treat as "the peer disconnected", not an error.
pub async fn read_message(stream: &mut TcpStream) -> Result<Option<ClusterMessage>> {
    let mut len_buf = [0u8; 4];
    match stream.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e).context("failed to read message length"),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_MESSAGE_BYTES {
        bail!(
            "cluster message declares {len} bytes, exceeding the {MAX_MESSAGE_BYTES}-byte cap - \
             rejecting before allocating (malformed peer or protocol mismatch)"
        );
    }
    let mut payload = vec![0u8; len as usize];
    stream.read_exact(&mut payload).await.context("failed to read message body (peer disconnected mid-message)")?;
    let msg: ClusterMessage = serde_json::from_slice(&payload).context("failed to parse cluster message JSON")?;
    Ok(Some(msg))
}
