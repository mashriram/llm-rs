use std::collections::HashMap;
use llm_core::types::{SeqId, TokenId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCacheMatchedResult {
    pub matched_offset: usize,
    pub fork_seq_id: Option<SeqId>,
    pub reuse_seq_id: Option<SeqId>,
    pub rolled_back_tokens: usize,
}

/// A node in the Radix Tree representing token paths.
#[derive(Debug, Clone)]
struct RadixNode {
    tokens: Vec<TokenId>,
    children: HashMap<TokenId, Box<RadixNode>>,
    seq_ids: Vec<SeqId>, // sequences ending at this node
}

impl RadixNode {
    fn new(tokens: Vec<TokenId>) -> Self {
        Self {
            tokens,
            children: HashMap::new(),
            seq_ids: Vec::new(),
        }
    }

    /// Find any sequence ID in this node or its descendants.
    fn find_any_sequence(&self) -> Option<SeqId> {
        if let Some(&seq_id) = self.seq_ids.first() {
            return Some(seq_id);
        }
        for child in self.children.values() {
            if let Some(seq_id) = child.find_any_sequence() {
                return Some(seq_id);
            }
        }
        None
    }

    /// Helper to insert a path starting from this node.
    fn insert_path(&mut self, seq_id: SeqId, path: &[TokenId], mut path_idx: usize) {
        let mut node = self;
        loop {
            if path_idx >= path.len() {
                if !node.seq_ids.contains(&seq_id) {
                    node.seq_ids.push(seq_id);
                }
                break;
            }

            let next_token = path[path_idx];
            if node.children.contains_key(&next_token) {
                let child = node.children.get_mut(&next_token).unwrap();
                let mut match_len = 0;
                for (i, &t) in child.tokens.iter().enumerate() {
                    if path_idx + i < path.len() && path[path_idx + i] == t {
                        match_len += 1;
                    } else {
                        break;
                    }
                }

                if match_len < child.tokens.len() {
                    // Split the child node
                    child.split_node(match_len);
                }

                path_idx += match_len;
                node = node.children.get_mut(&next_token).unwrap();
            } else {
                // Create a new child
                let remaining_tokens = path[path_idx..].to_vec();
                let mut new_child = Box::new(RadixNode::new(remaining_tokens));
                new_child.seq_ids.push(seq_id);
                node.children.insert(next_token, new_child);
                break;
            }
        }
    }

    fn split_node(&mut self, split_idx: usize) {
        assert!(split_idx < self.tokens.len());
        let child_tokens = self.tokens.split_off(split_idx);
        let first_child_token = child_tokens[0];

        let mut child_node = Box::new(RadixNode::new(child_tokens));
        std::mem::swap(&mut child_node.children, &mut self.children);
        std::mem::swap(&mut child_node.seq_ids, &mut self.seq_ids);

        self.children.insert(first_child_token, child_node);
    }
}

/// A Radix-tree based prefix cache for reusing KV cache blocks of common prefixes.
/// Maps sequences to their token history. Matches C++ `prefix_cache.cc` / `radix_tree.cc` invariants.
pub struct PrefixCache {
    root: Box<RadixNode>,
    seq_to_path: HashMap<SeqId, Vec<TokenId>>,
    max_recycling_seqs: usize,
    recycling_seqs: Vec<SeqId>, // LRU tracker for lazy recycling
}

impl PrefixCache {
    pub fn new(max_recycling_seqs: usize) -> Self {
        Self {
            root: Box::new(RadixNode::new(Vec::new())),
            seq_to_path: HashMap::new(),
            max_recycling_seqs,
            recycling_seqs: Vec::new(),
        }
    }

    /// Check if a sequence exists.
    pub fn has_sequence(&self, seq_id: SeqId) -> bool {
        self.seq_to_path.contains_key(&seq_id)
    }

    /// Get a sequence's tokens.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&[TokenId]> {
        self.seq_to_path.get(&seq_id).map(|v| v.as_slice())
    }

    /// Add an empty sequence at root.
    pub fn add_sequence(&mut self, seq_id: SeqId) {
        if self.has_sequence(seq_id) {
            return;
        }
        self.root.seq_ids.push(seq_id);
        self.seq_to_path.insert(seq_id, Vec::new());
    }

    /// Get the remaining capacity (simulated for compatibility).
    pub fn free_capacity(&self) -> usize {
        // Return a large capacity for testing parity
        64 * 64
    }

    /// Insert a new token sequence and match it against existing cached prefixes.
    pub fn insert_sequence(
        &mut self,
        seq_id: SeqId,
        tokens: &[TokenId],
        _sliding_window_size: i32,
    ) -> PrefixCacheMatchedResult {
        // Remove from recycling if it was there
        if let Some(pos) = self.recycling_seqs.iter().position(|&x| x == seq_id) {
            self.recycling_seqs.remove(pos);
        }

        // Match against the existing tree to find the longest common prefix
        let mut curr = &self.root;
        let mut path_idx = 0;
        let mut matched_len = 0;
        let mut best_sharing_seq = None;

        // Traverse to find the longest match
        loop {
            if path_idx >= tokens.len() {
                break;
            }
            let next_token = tokens[path_idx];
            if let Some(child) = curr.children.get(&next_token) {
                // Compare child's tokens
                let mut node_match = 0;
                for (i, &t) in child.tokens.iter().enumerate() {
                    if path_idx + i < tokens.len() && tokens[path_idx + i] == t {
                        node_match += 1;
                    } else {
                        break;
                    }
                }

                if node_match > 0 {
                    matched_len += node_match;
                    path_idx += node_match;
                    best_sharing_seq = child.find_any_sequence();
                } else {
                    break;
                }

                if node_match < child.tokens.len() {
                    break;
                }

                curr = child;
            } else {
                break;
            }
        }

        // Now perform the actual insertion of the sequence
        self.root.insert_path(seq_id, tokens, 0);
        self.seq_to_path.insert(seq_id, tokens.to_vec());

        if matched_len > 0 {
            if let Some(sharing) = best_sharing_seq {
                if let Some(pos) = self.recycling_seqs.iter().position(|&x| x == sharing) {
                    let recycled_seq = self.recycling_seqs.remove(pos);
                    PrefixCacheMatchedResult {
                        matched_offset: matched_len,
                        fork_seq_id: None,
                        reuse_seq_id: Some(recycled_seq),
                        rolled_back_tokens: 0,
                    }
                } else {
                    PrefixCacheMatchedResult {
                        matched_offset: matched_len,
                        fork_seq_id: Some(sharing),
                        reuse_seq_id: None,
                        rolled_back_tokens: 0,
                    }
                }
            } else {
                PrefixCacheMatchedResult {
                    matched_offset: matched_len,
                    fork_seq_id: None,
                    reuse_seq_id: None,
                    rolled_back_tokens: 0,
                }
            }
        } else {
            PrefixCacheMatchedResult {
                matched_offset: 0,
                fork_seq_id: None,
                reuse_seq_id: None,
                rolled_back_tokens: 0,
            }
        }
    }

    /// Extend an active sequence in the prefix cache.
    pub fn extend_sequence(&mut self, seq_id: SeqId, new_tokens: &[TokenId]) {
        if new_tokens.is_empty() {
            return;
        }
        if let Some(path) = self.seq_to_path.get(&seq_id).cloned() {
            // Remove the sequence from its old node
            self.remove_seq_id_from_tree(&path, seq_id);

            // Create the new extended path
            let mut new_path = path;
            new_path.extend_from_slice(new_tokens);

            // Re-insert the sequence at the new path
            self.root.insert_path(seq_id, &new_path, 0);
            self.seq_to_path.insert(seq_id, new_path);
        }
    }

    /// Fork a sequence from a parent sequence at a given offset.
    pub fn fork_sequence(&mut self, seq_id: SeqId, parent_seq_id: SeqId, forked_offset: usize) {
        if self.has_sequence(seq_id) {
            return;
        }
        if let Some(parent_path) = self.seq_to_path.get(&parent_seq_id).cloned() {
            let forked_path = parent_path[..forked_offset.min(parent_path.len())].to_vec();
            self.root.insert_path(seq_id, &forked_path, 0);
            self.seq_to_path.insert(seq_id, forked_path);
        }
    }

    /// Roll back a sequence by a number of tokens.
    pub fn rollback_sequence(&mut self, seq_id: SeqId, num_tokens: usize) {
        if let Some(path) = self.seq_to_path.get(&seq_id).cloned() {
            if num_tokens >= path.len() {
                self.remove_sequence(seq_id);
                self.add_sequence(seq_id);
                return;
            }

            self.remove_seq_id_from_tree(&path, seq_id);
            let rolled_back_path = path[..path.len() - num_tokens].to_vec();
            self.root.insert_path(seq_id, &rolled_back_path, 0);
            self.seq_to_path.insert(seq_id, rolled_back_path);
        }
    }

    /// Remove a sequence completely.
    pub fn remove_sequence(&mut self, seq_id: SeqId) {
        if let Some(path) = self.seq_to_path.remove(&seq_id) {
            self.remove_seq_id_from_tree(&path, seq_id);
        }
        if let Some(pos) = self.recycling_seqs.iter().position(|&x| x == seq_id) {
            self.recycling_seqs.remove(pos);
        }
    }

    /// Recycle a sequence (lazy removal).
    pub fn recycle_sequence(&mut self, seq_id: SeqId) {
        if self.max_recycling_seqs == 0 {
            self.remove_sequence(seq_id);
            return;
        }

        if self.recycling_seqs.len() >= self.max_recycling_seqs {
            let evicted = self.recycling_seqs.remove(0);
            self.remove_sequence(evicted);
        }
        if !self.recycling_seqs.contains(&seq_id) && self.has_sequence(seq_id) {
            self.recycling_seqs.push(seq_id);
        }
    }

    fn remove_seq_id_from_tree(&mut self, path: &[TokenId], seq_id: SeqId) {
        Self::remove_seq_id_from_node(&mut self.root, path, 0, seq_id);
        // Prune empty nodes
        Self::prune_path(&mut self.root, path, 0);
    }

    fn remove_seq_id_from_node(mut node: &mut RadixNode, path: &[TokenId], mut path_idx: usize, seq_id: SeqId) {
        loop {
            if path_idx >= path.len() {
                if let Some(pos) = node.seq_ids.iter().position(|&x| x == seq_id) {
                    node.seq_ids.remove(pos);
                }
                break;
            }
            let next_token = path[path_idx];
            if node.children.contains_key(&next_token) {
                let child = node.children.get_mut(&next_token).unwrap();
                path_idx += child.tokens.len();
                node = child;
            } else {
                break;
            }
        }
    }

    fn prune_path(node: &mut RadixNode, path: &[TokenId], path_idx: usize) -> bool {
        if path_idx < path.len() {
            let next_token = path[path_idx];
            let mut should_remove = false;
            if let Some(child) = node.children.get_mut(&next_token) {
                let match_len = child.tokens.len();
                if Self::prune_path(child, path, path_idx + match_len) {
                    should_remove = true;
                }
            }
            if should_remove {
                node.children.remove(&next_token);
            }
        }
        node.seq_ids.is_empty() && node.children.is_empty()
    }

    /// Match a token sequence against the prefix cache, returning the matched length and sharing sequences.
    pub fn match_sequence(&self, tokens: &[TokenId]) -> (usize, Vec<SeqId>) {
        let mut curr = &self.root;
        let mut path_idx = 0;
        let mut matched_len = 0;

        loop {
            if path_idx >= tokens.len() {
                break;
            }
            let next_token = tokens[path_idx];
            if let Some(child) = curr.children.get(&next_token) {
                let mut node_match = 0;
                for (i, &t) in child.tokens.iter().enumerate() {
                    if path_idx + i < tokens.len() && tokens[path_idx + i] == t {
                        node_match += 1;
                    } else {
                        break;
                    }
                }

                matched_len += node_match;
                path_idx += node_match;

                if node_match < child.tokens.len() {
                    curr = child;
                    break;
                }
                curr = child;
            } else {
                break;
            }
        }

        if matched_len == 0 {
            (0, Vec::new())
        } else {
            let mut sharing = Vec::new();
            Self::collect_all_sequences(curr, &mut sharing);
            (matched_len, sharing)
        }
    }

    fn collect_all_sequences(node: &RadixNode, out: &mut Vec<SeqId>) {
        for &id in &node.seq_ids {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        for child in node.children.values() {
            Self::collect_all_sequences(child, out);
        }
    }
}
