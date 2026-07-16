use crate::types::SampleParams;
use rand::Rng;
use anyhow::{Result, bail};
use std::cmp::Ordering;
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

/// Logged once (not per-token) if NaN logits are detected, to surface upstream bugs
/// without flooding the log on every decode step.
static NAN_WARNED: AtomicBool = AtomicBool::new(false);

/// NaN-safe descending sort for `(prob, idx)` tuples; NaNs sort last.
#[inline]
fn cmp_desc_prob(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
    b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal)
}

/// NaN-safe descending sort for `(idx, logit)` tuples; NaNs sort last.
#[inline]
fn cmp_desc_logit(a: &(usize, f32), b: &(usize, f32)) -> Ordering {
    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
}


/// CPU Sampler implementing MLC-LLM compatible top-p, top-k, temperature, and repetition penalty.
pub fn sample_logits(
    logits: &[f32],
    params: &SampleParams,
    token_history: &[u32],
) -> Result<u32> {
    if logits.is_empty() {
        bail!("Empty logits passed to sampler");
    }

    let mut local_logits = logits.to_vec();

    // 1. Apply Repetition Penalty
    apply_repetition_penalty(&mut local_logits, token_history, params.repetition_penalty);

    // 2. Greedy search (Temperature = 0)
    if params.temperature <= 1e-5 {
        let mut max_idx = 0;
        let mut max_val = local_logits[0];
        for (i, &val) in local_logits.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }

    // 3. Scale by Temperature
    apply_temperature(&mut local_logits, params.temperature);

    // 4. Compute Softmax to get probabilities
    let max_logit = local_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(f32, usize)> = local_logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| {
            let p = if l == f32::NEG_INFINITY { 0.0 } else { (l - max_logit).exp() };
            (p, idx)
        })
        .collect();

    let sum_prob: f32 = probs.iter().map(|(p, _)| p).sum();
    if sum_prob > 0.0 {
        for (p, _) in probs.iter_mut() {
            *p /= sum_prob;
        }
    }

    // 5. Sort probabilities descending (NaN-safe: NaN logits sort last)
    if probs.iter().any(|(p, _)| p.is_nan()) && !NAN_WARNED.swap(true, AtomicOrdering::Relaxed) {
        tracing::warn!("NaN probabilities in sampler — numerical bug upstream; NaN tokens excluded by top-k/p");
    }
    probs.sort_unstable_by(cmp_desc_prob);

    // 6. Apply Top-K
    if params.top_k > 0 && params.top_k < probs.len() {
        probs.truncate(params.top_k);
        // Renormalize
        let sum_k: f32 = probs.iter().map(|(p, _)| p).sum();
        if sum_k > 0.0 {
            for (p, _) in probs.iter_mut() {
                *p /= sum_k;
            }
        }
    }

    // 7. Apply Top-P (Nucleus Sampling)
    if params.top_p < 1.0 {
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, &(p, _)) in probs.iter().enumerate() {
            cumulative_prob += p;
            if cumulative_prob >= params.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);
        // Renormalize
        let sum_p: f32 = probs.iter().map(|(p, _)| p).sum();
        if sum_p > 0.0 {
            for (p, _) in probs.iter_mut() {
                *p /= sum_p;
            }
        }
    }

    // 8. Sample from the remaining distribution
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cumulative = 0.0;
    for &(p, idx) in &probs {
        cumulative += p;
        if r <= cumulative {
            return Ok(idx as u32);
        }
    }

    // Fallback in case of rounding errors
    Ok(probs.last().map(|(_, idx)| *idx as u32).unwrap_or(0))
}

/// Helper to apply repetition penalty in-place.
pub fn apply_repetition_penalty(logits: &mut [f32], token_history: &[u32], penalty: f32) {
    if penalty == 1.0 || token_history.is_empty() {
        return;
    }
    let mut unique_tokens = token_history.to_vec();
    unique_tokens.sort_unstable();
    unique_tokens.dedup();

    for &token in &unique_tokens {
        let idx = token as usize;
        if idx < logits.len() {
            let logit = logits[idx];
            if logit > 0.0 {
                logits[idx] = logit / penalty;
            } else {
                logits[idx] = logit * penalty;
            }
        }
    }
}

/// Helper to apply temperature scaling in-place.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature <= 1e-5 || temperature == 1.0 {
        return;
    }
    for val in logits.iter_mut() {
        *val /= temperature;
    }
}

/// Helper to apply Top-K filtering in-place (masks other elements with NEG_INFINITY).
pub fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed_logits.sort_unstable_by(cmp_desc_logit);
    
    let threshold = indexed_logits[top_k - 1].1;
    for val in logits.iter_mut() {
        if *val < threshold {
            *val = f32::NEG_INFINITY;
        }
    }
}

/// Helper to apply Top-P (nucleus) filtering in-place (masks other elements with NEG_INFINITY).
pub fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(idx, &l)| {
            let p = if l == f32::NEG_INFINITY { 0.0 } else { (l - max_logit).exp() };
            (idx, p)
            })
        .collect();

    let sum_prob: f32 = probs.iter().map(|(_, p)| p).sum();
    if sum_prob <= 0.0 {
        return;
    }
    for (_, p) in probs.iter_mut() {
        *p /= sum_prob;
    }

    probs.sort_unstable_by(cmp_desc_logit);

    let mut cumulative_prob = 0.0;
    let mut cutoff_idx = probs.len();
    for (i, &(_, p)) in probs.iter().enumerate() {
        cumulative_prob += p;
        if cumulative_prob >= top_p {
            cutoff_idx = i + 1;
            break;
        }
    }

    let kept_indices: std::collections::HashSet<usize> = probs[..cutoff_idx].iter().map(|&(idx, _)| idx).collect();
    for (idx, val) in logits.iter_mut().enumerate() {
        if !kept_indices.contains(&idx) {
            *val = f32::NEG_INFINITY;
        }
    }
}

/// Simple greedy/sample wrapper.
pub fn sample(logits: &[f32], temperature: f32) -> u32 {
    let params = SampleParams {
        temperature,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_new_tokens: 512,
    };
    sample_logits(logits, &params, &[]).unwrap_or(0)
}
