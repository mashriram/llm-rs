use std::collections::VecDeque;
use anyhow::{Result, bail};
use llm_core::types::BlockId;

/// Manages a pool of physical blocks for PagedAttention.
/// Tracks free blocks and reference counts for shared blocks (e.g. parallel generation, prefix cache).
#[derive(Debug)]
pub struct BlockAllocator {
    capacity: usize,
    free_list: VecDeque<BlockId>,
    ref_counts: Vec<u32>,
}

impl BlockAllocator {
    pub fn new(capacity: usize) -> Self {
        let mut free_list = VecDeque::with_capacity(capacity);
        for i in 0..capacity {
            free_list.push_back(i as BlockId);
        }
        Self {
            capacity,
            free_list,
            ref_counts: vec![0; capacity],
        }
    }

    /// Allocate a single block.
    pub fn allocate(&mut self) -> Result<BlockId> {
        if let Some(block_id) = self.free_list.pop_front() {
            self.ref_counts[block_id as usize] = 1;
            Ok(block_id)
        } else {
            bail!("Out of physical blocks in the PagedAttention block pool");
        }
    }

    /// Increment the reference count of a block (shared block).
    pub fn increment_ref(&mut self, block_id: BlockId) -> Result<()> {
        let idx = block_id as usize;
        if idx >= self.capacity {
            bail!("Invalid block ID: {}", block_id);
        }
        if self.ref_counts[idx] == 0 {
            bail!("Cannot increment ref count of an unallocated block: {}", block_id);
        }
        self.ref_counts[idx] += 1;
        Ok(())
    }

    /// Decrement the reference count of a block.
    /// If the reference count drops to 0, the block is returned to the free list.
    /// Returns `true` if the block was freed.
    pub fn free(&mut self, block_id: BlockId) -> Result<bool> {
        let idx = block_id as usize;
        if idx >= self.capacity {
            bail!("Invalid block ID: {}", block_id);
        }
        if self.ref_counts[idx] == 0 {
            bail!("Double free detected for block: {}", block_id);
        }
        
        self.ref_counts[idx] -= 1;
        if self.ref_counts[idx] == 0 {
            self.free_list.push_back(block_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get the number of free blocks remaining.
    pub fn free_blocks(&self) -> usize {
        self.free_list.len()
    }

    /// Get the total block capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
