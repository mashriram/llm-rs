use tokio::sync::mpsc::{unbounded_channel, UnboundedSender, UnboundedReceiver};
use tokio::sync::broadcast;
use anyhow::Result;
use tracing::{info, error};

use llm_core::backend::LlmBackend;
use llm_core::types::{InferRequest, TokenId, SeqId};
use crate::scheduler::Scheduler;

/// Event sent from the serving engine to subscribers when a token is generated.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub seq_id: SeqId,
    pub token_id: TokenId,
    pub is_eos: bool,
}

/// Thread-safe orchestrator for the LLM inference engine.
/// Spawns a background task that drives the scheduler step loop.
pub struct ServingEngine {
    request_tx: UnboundedSender<InferRequest>,
    event_tx: broadcast::Sender<TokenEvent>,
}

impl ServingEngine {
    pub fn new(backend: Box<dyn LlmBackend>, block_pool_size: usize) -> Self {
        let (request_tx, request_rx) = unbounded_channel();
        let (event_tx, _) = broadcast::channel(1024);

        let event_tx_clone = event_tx.clone();
        tokio::spawn(async move {
            if let Err(e) = Self::run_engine_loop(backend, block_pool_size, request_rx, event_tx_clone).await {
                error!("ServingEngine background loop failed: {:?}", e);
            }
        });

        Self {
            request_tx,
            event_tx,
        }
    }

    /// Add a request to the serving engine.
    pub fn add_request(&self, request: InferRequest) -> Result<()> {
        self.request_tx.send(request)
            .map_err(|_| anyhow::anyhow!("Failed to send request to ServingEngine background task"))
    }

    /// Subscribe to the token generation events.
    pub fn subscribe(&self) -> broadcast::Receiver<TokenEvent> {
        self.event_tx.subscribe()
    }

    async fn run_engine_loop(
        backend: Box<dyn LlmBackend>,
        block_pool_size: usize,
        mut request_rx: UnboundedReceiver<InferRequest>,
        event_tx: broadcast::Sender<TokenEvent>,
    ) -> Result<()> {
        let mut scheduler = Scheduler::new(backend, block_pool_size);
        info!("ServingEngine background loop started.");

        loop {
            // 1. Handle incoming requests
            if scheduler.running_tasks() == 0 && scheduler.waiting_tasks() == 0 {
                // Blocking wait if there are no active or waiting sequences
                match request_rx.recv().await {
                    Some(req) => {
                        scheduler.add_request(req);
                    }
                    None => {
                        // Channel closed, shutdown
                        break;
                    }
                }
            } else {
                // Non-blocking poll for new requests when we are already processing batches
                while let Ok(req) = request_rx.try_recv() {
                    scheduler.add_request(req);
                }
            }

            // 2. Step the scheduler
            if scheduler.running_tasks() > 0 || scheduler.waiting_tasks() > 0 {
                match scheduler.step() {
                    Ok(results) => {
                        for (seq_id, token_id, is_eos) in results {
                            let event = TokenEvent { seq_id, token_id, is_eos };
                            let _ = event_tx.send(event);
                        }
                    }
                    Err(e) => {
                        error!("Error during scheduler step: {:?}", e);
                        scheduler.abort_all_running();
                    }
                }

                // Yield to the executor to prevent starving other tasks in the async runtime
                tokio::task::yield_now().await;
            }
        }

        info!("ServingEngine background loop shutting down.");
        Ok(())
    }
}
