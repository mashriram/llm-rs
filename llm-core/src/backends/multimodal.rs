//! Multimodal (image) embedding helpers.
//!
//! Handles visual embedding splicing into the text embedding sequence.
//! Supports two detection strategies:
//! 1. Run-based: longest consecutive run of the same token ID (>=16 tokens)
//!    is assumed to be the image placeholder region.
//! 2. Marker-based: explicit vision_start_id / vision_end_id tokens.

use anyhow::Result;
use candle_core::Tensor;

/// Splice pre-computed visual embeddings into the text embedding tensor
/// at the image placeholder position.
///
/// The spliced tensor has the same hidden dimension as the input but its
/// sequence length reflects the visual token count if it differs from the
/// number of placeholder tokens (it is clamped or padded).
///
/// # Arguments
/// * `text_embeds`    — `[1, seq_len, hidden_dim]`
/// * `visual_embeds`  — `[1, n_visual_tokens, hidden_dim]`
/// * `token_ids`      — raw input token IDs used for placeholder detection
/// * `vision_start_id` / `vision_end_id` — model-specific marker token IDs
pub(crate) fn splice_visual_embeddings(
    text_embeds: &Tensor,
    visual_embeds: &Tensor,
    token_ids: &[u32],
    vision_start_id: u32,
    vision_end_id: u32,
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_dim) = text_embeds.dims3()?;
    // Multimodal splicing is only supported for batch_size == 1 during prefill.
    if b_sz != 1 {
        return Ok(text_embeds.clone());
    }

    // --- Strategy 1: longest consecutive run (model-agnostic) ---
    let (max_run_len, max_run_start) = find_longest_run(token_ids);
    if max_run_len >= 16 {
        return splice_at(text_embeds, visual_embeds, max_run_start, max_run_len, seq_len, hidden_dim);
    }

    // --- Strategy 2: explicit start/end marker tokens ---
    let mut start_idx = None;
    let mut end_idx = None;
    for (idx, &tok) in token_ids.iter().enumerate() {
        if tok == vision_start_id {
            start_idx = Some(idx);
        } else if tok == vision_end_id {
            end_idx = Some(idx);
        }
    }

    if let (Some(start), Some(end)) = (start_idx, end_idx) {
        if end > start + 1 {
            let num_pads = end - start - 1;
            return splice_at(text_embeds, visual_embeds, start + 1, num_pads, seq_len, hidden_dim);
        }
    }

    // No placeholder found — return unmodified.
    Ok(text_embeds.clone())
}

/// Find the start index and length of the longest run of identical token IDs.
fn find_longest_run(token_ids: &[u32]) -> (usize, usize) {
    if token_ids.is_empty() {
        return (0, 0);
    }
    let mut max_len = 0;
    let mut max_start = 0;
    let mut cur_len = 1;
    let mut cur_start = 0;

    for i in 1..token_ids.len() {
        if token_ids[i] == token_ids[i - 1] {
            cur_len += 1;
        } else {
            if cur_len > max_len {
                max_len = cur_len;
                max_start = cur_start;
            }
            cur_len = 1;
            cur_start = i;
        }
    }
    if cur_len > max_len {
        max_len = cur_len;
        max_start = cur_start;
    }
    (max_len, max_start)
}

/// Replace `[run_start .. run_start + run_len]` in `text_embeds` with
/// the visual embedding tensor, padding/clipping as needed.
fn splice_at(
    text_embeds: &Tensor,
    visual_embeds: &Tensor,
    run_start: usize,
    run_len: usize,
    seq_len: usize,
    hidden_dim: usize,
) -> Result<Tensor> {
    let visual_len = visual_embeds.dim(1)?;

    let before = text_embeds.narrow(1, 0, run_start)?;
    let after_start = run_start + run_len;
    let after_len = seq_len.saturating_sub(after_start);
    let after = text_embeds.narrow(1, after_start, after_len)?;

    let middle = if visual_len == run_len {
        visual_embeds.clone()
    } else if visual_len > run_len {
        // Clip to placeholder length.
        visual_embeds.narrow(1, 0, run_len)?
    } else {
        // Pad with zeros to fill the placeholder.
        let pad_len = run_len - visual_len;
        let pad = Tensor::zeros(
            (1, pad_len, hidden_dim),
            visual_embeds.dtype(),
            visual_embeds.device(),
        )?;
        Tensor::cat(&[visual_embeds, &pad], 1)?
    };

    // Cast visual embeddings to match text embedding dtype (e.g. f32 vs f16).
    let middle = middle.to_dtype(text_embeds.dtype())?;

    Ok(Tensor::cat(&[&before, &middle, &after], 1)?)
}

/// Splice pre-computed audio embeddings into the text embedding tensor
/// at the audio placeholder position.
pub(crate) fn splice_audio_embeddings(
    text_embeds: &Tensor,
    audio_embeds: &Tensor,
    token_ids: &[u32],
) -> Result<Tensor> {
    let (b_sz, seq_len, hidden_dim) = text_embeds.dims3()?;
    if b_sz != 1 {
        return Ok(text_embeds.clone());
    }

    // Since chat.rs expands <audio> into 750 tokens of either <|audio_pad|> or <|audio|> etc.,
    // the longest run of identical tokens in the sequence will represent the audio placeholder.
    let (max_run_len, max_run_start) = find_longest_run(token_ids);
    if max_run_len >= 16 {
        return splice_at(text_embeds, audio_embeds, max_run_start, max_run_len, seq_len, hidden_dim);
    }

    Ok(text_embeds.clone())
}
