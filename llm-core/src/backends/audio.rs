use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, Context};
use candle_core::{Tensor, Device, DType};
use symphonia::core::audio::Signal;

/// Which audio-encoder architecture a loaded checkpoint uses. Detected from
/// tensor names present in the checkpoint at load time (see
/// `detect_architecture`) — never user-specified, per the model-agnostic
/// design: any supported checkpoint should just work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioArchitecture {
    /// Gemma-4 style Conformer encoder (SubSampleConvProjection + chunked
    /// local attention with relative position bias). Native GGUF tensor
    /// names: `a.conv1d.*`, `a.blk.*`.
    GemmaConformer,
    /// Whisper-style encoder (Conv1d x2 + absolute positional embedding +
    /// standard MHA transformer blocks). Tensor names:
    /// `audio_encoder.*`, `audio_projector.*` (several naming variants
    /// normalized by `normalize_audio_tensors`).
    Whisper,
}

impl AudioArchitecture {
    /// Expected number of mel-spectrogram bins for this architecture's
    /// feature extractor. Used by `load_audio` to build the right input
    /// shape — Whisper checkpoints expect 80 bins, Gemma-4's 128.
    pub fn num_mel_bins(self) -> usize {
        match self {
            AudioArchitecture::GemmaConformer => 128,
            AudioArchitecture::Whisper => 80,
        }
    }

    fn defaults(self) -> (usize, usize, usize, usize) {
        // (hidden_dim, num_layers, num_heads, projection_dim)
        match self {
            AudioArchitecture::GemmaConformer => (1024, 12, 8, 1024),
            AudioArchitecture::Whisper => (1280, 32, 20, 2560),
        }
    }
}

/// Inspect raw (un-normalized) tensor names to decide which encoder
/// architecture a checkpoint implements. Whisper-derived checkpoints use
/// `audio_encoder.*`/`audio_projector.*` naming; Gemma-4's GGUF export uses
/// `a.conv1d.*`/`a.blk.*` directly, so absence of Whisper naming defaults to
/// GemmaConformer.
fn detect_architecture(weights: &HashMap<String, Tensor>) -> AudioArchitecture {
    let is_whisper = weights.keys().any(|k| {
        k.contains("audio_encoder.") || k.contains("audio_projector.")
    });
    if is_whisper {
        AudioArchitecture::Whisper
    } else {
        AudioArchitecture::GemmaConformer
    }
}

pub struct AudioEncoder {
    weights: HashMap<String, Tensor>,
    pub architecture: AudioArchitecture,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub projection_dim: usize,
}

impl AudioEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!("Loading audio encoder from {:?}", path);

        let mut raw_weights = HashMap::new();
        let mut hidden_dim_override: Option<usize> = None;
        let mut num_layers_override: Option<usize> = None;
        let mut num_heads_override: Option<usize> = None;
        let mut projection_dim_override: Option<usize> = None;

        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        if is_gguf {
            let mut file = std::fs::File::open(path)
                .context(format!("Failed to open audio GGUF file: {:?}", path))?;
            let model = candle_core::quantized::gguf_file::Content::read(&mut file)
                .context("Failed to read GGUF audio content")?;

            let get_metadata_u32 = |key: &str| -> Option<u32> {
                match model.metadata.get(key) {
                    Some(candle_core::quantized::gguf_file::Value::U32(v)) => Some(*v),
                    Some(candle_core::quantized::gguf_file::Value::I32(v)) => Some(*v as u32),
                    _ => None,
                }
            };

            if let Some(v) = get_metadata_u32("clip.audio.embedding_length") { hidden_dim_override = Some(v as usize); }
            if let Some(v) = get_metadata_u32("clip.audio.block_count") { num_layers_override = Some(v as usize); }
            if let Some(v) = get_metadata_u32("clip.audio.attention.head_count") { num_heads_override = Some(v as usize); }
            if let Some(v) = get_metadata_u32("clip.audio.projection_dim") { projection_dim_override = Some(v as usize); }

            let cpu = Device::Cpu;
            let audio_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            for name in model.tensor_infos.keys() {
                let qtensor = model.tensor(&mut file, name, &cpu)
                    .context(format!("Failed to load audio tensor {}", name))?;
                let tensor = qtensor.dequantize(&cpu)
                    .context(format!("Failed to dequantize audio tensor {}", name))?
                    .to_dtype(audio_dtype)?
                    .to_device(device)?;
                raw_weights.insert(name.clone(), tensor);
            }
        } else {
            // Safetensors
            let loaded = if path.is_dir() {
                let mut sfs = HashMap::new();
                for entry in std::fs::read_dir(path)?.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
                        let chunk = candle_core::safetensors::load(&p, device)?;
                        sfs.extend(chunk);
                    }
                }
                sfs
            } else {
                candle_core::safetensors::load(path, device)?
            };

            let audio_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            for (k, v) in loaded {
                raw_weights.insert(k, v.to_dtype(audio_dtype)?);
            }
        }

        let architecture = detect_architecture(&raw_weights);
        let (default_hidden, default_layers, default_heads, default_proj) = architecture.defaults();
        let hidden_dim = hidden_dim_override.unwrap_or(default_hidden);
        let num_layers = num_layers_override.unwrap_or(default_layers);
        let num_heads = num_heads_override.unwrap_or(default_heads);
        let projection_dim = projection_dim_override.unwrap_or(default_proj);

        tracing::info!(
            "Detected audio encoder architecture: {:?} (hidden_dim={}, num_layers={}, num_heads={})",
            architecture, hidden_dim, num_layers, num_heads
        );

        let weights = match architecture {
            AudioArchitecture::Whisper => normalize_audio_tensors(raw_weights, num_layers, device)?,
            AudioArchitecture::GemmaConformer => raw_weights,
        };

        Ok(Self {
            weights,
            architecture,
            hidden_dim,
            num_layers,
            num_heads,
            projection_dim,
        })
    }

    /// `valid_frames`: how many of `audio_values`' time-axis positions are
    /// real audio vs. zero-padding (see `load_audio`). Only used by the
    /// Gemma-Conformer path to mask padded positions out of attention/
    /// normalization; ignored by Whisper (see `load_audio`'s doc comment
    /// for why that path isn't masked).
    pub fn encode(&self, audio_values: &Tensor, valid_frames: usize) -> Result<Tensor> {
        match self.architecture {
            AudioArchitecture::GemmaConformer => self.encode_conformer(audio_values, valid_frames),
            AudioArchitecture::Whisper => self.encode_whisper(audio_values),
        }
    }

    /// Whisper-style encoder: Conv1d x2 -> absolute positional embedding ->
    /// standard MHA transformer blocks -> post-LN -> 2-layer MLP projector.
    fn encode_whisper(&self, audio_values: &Tensor) -> Result<Tensor> {
        let conv1_w = self.weights.get("audio.conv1.weight")
            .ok_or_else(|| anyhow!("audio.conv1.weight not found"))?;
        let conv1_b = self.weights.get("audio.conv1.bias");

        let conv2_w = self.weights.get("audio.conv2.weight")
            .ok_or_else(|| anyhow!("audio.conv2.weight not found"))?;
        let conv2_b = self.weights.get("audio.conv2.bias");

        // Conv1: stride 1, padding 1
        let mut x = audio_values.conv1d(conv1_w, 1, 1, 1, 1)?;
        if let Some(bias) = conv1_b {
            let b = bias.reshape((1, bias.dim(0)?, 1))?;
            x = x.broadcast_add(&b)?;
        }
        x = x.gelu()?;

        // Conv2: stride 2, padding 1
        x = x.conv1d(conv2_w, 2, 1, 1, 1)?;
        if let Some(bias) = conv2_b {
            let b = bias.reshape((1, bias.dim(0)?, 1))?;
            x = x.broadcast_add(&b)?;
        }
        x = x.gelu()?;

        // Transpose to [batch, seq_len, hidden_dim]
        x = x.transpose(1, 2)?;

        // Positional embeddings
        if let Some(pos_emb) = self.weights.get("audio.pos_embed.weight") {
            let seq_len = x.dim(1)?;
            let emb_len = pos_emb.dim(0)?;
            let sliced_emb = if seq_len < emb_len {
                pos_emb.narrow(0, 0, seq_len)?
            } else {
                pos_emb.clone()
            };
            x = x.broadcast_add(&sliced_emb)?;
        }

        // Transformer encoder blocks
        let head_dim = self.hidden_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        for i in 0..self.num_layers {
            let ln1_w = self.weights.get(&format!("audio.layers.{}.ln1.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ln1.weight not found", i))?;
            let ln1_b = self.weights.get(&format!("audio.layers.{}.ln1.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ln1.bias not found", i))?;
            let ln2_w = self.weights.get(&format!("audio.layers.{}.ln2.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ln2.weight not found", i))?;
            let ln2_b = self.weights.get(&format!("audio.layers.{}.ln2.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ln2.bias not found", i))?;

            let qkv_w = self.weights.get(&format!("audio.layers.{}.attn_qkv.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.attn_qkv.weight not found", i))?;
            let qkv_b = self.weights.get(&format!("audio.layers.{}.attn_qkv.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.attn_qkv.bias not found", i))?;

            let attn_out_w = self.weights.get(&format!("audio.layers.{}.attn_out.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.attn_out.weight not found", i))?;
            let attn_out_b = self.weights.get(&format!("audio.layers.{}.attn_out.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.attn_out.bias not found", i))?;

            let ffn_up_w = self.weights.get(&format!("audio.layers.{}.ffn_up.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ffn_up.weight not found", i))?;
            let ffn_up_b = self.weights.get(&format!("audio.layers.{}.ffn_up.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ffn_up.bias not found", i))?;
            let ffn_down_w = self.weights.get(&format!("audio.layers.{}.ffn_down.weight", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ffn_down.weight not found", i))?;
            let ffn_down_b = self.weights.get(&format!("audio.layers.{}.ffn_down.bias", i))
                .ok_or_else(|| anyhow!("audio.layers.{}.ffn_down.bias not found", i))?;

            // LN1
            let norm_x = candle_nn::ops::layer_norm(&x, ln1_w, ln1_b, 1e-5)?;

            // Attention
            let qkv = matmul_3d_2d(&norm_x, qkv_w)?;
            let qkv = qkv.broadcast_add(qkv_b)?;

            let (b, seq, three_d) = qkv.dims3()?;
            let d = three_d / 3;

            let q = qkv.narrow(2, 0, d)?;
            let k = qkv.narrow(2, d, d)?;
            let v = qkv.narrow(2, 2 * d, d)?;

            // Reshape for MHA: [b, h, seq, head_dim]
            let q = q.reshape((b, seq, self.num_heads, head_dim))?.transpose(1, 2)?;
            let k = k.reshape((b, seq, self.num_heads, head_dim))?.transpose(1, 2)?;
            let v = v.reshape((b, seq, self.num_heads, head_dim))?.transpose(1, 2)?;

            let scores = q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)?;
            let scores = (scores * scale)?;
            let attn = candle_nn::ops::softmax(&scores, 3)?;
            let context = attn.matmul(&v.contiguous()?)?;

            // Transpose back: [b, seq, hidden_dim]
            let context = context.transpose(1, 2)?.contiguous()?.reshape((b, seq, d))?;
            let attn_out = matmul_3d_2d(&context, attn_out_w)?;
            let attn_out = attn_out.broadcast_add(attn_out_b)?;

            // Residual
            x = x.add(&attn_out)?;

            // LN2 + FFN
            let norm_x2 = candle_nn::ops::layer_norm(&x, ln2_w, ln2_b, 1e-5)?;
            let ffn = matmul_3d_2d(&norm_x2, ffn_up_w)?;
            let ffn = ffn.broadcast_add(ffn_up_b)?;
            let ffn = ffn.gelu()?;
            let ffn = matmul_3d_2d(&ffn, ffn_down_w)?;
            let ffn = ffn.broadcast_add(ffn_down_b)?;

            // Residual
            x = x.add(&ffn)?;
        }

        // Post LN
        if let (Some(post_ln_w), Some(post_ln_b)) = (self.weights.get("audio.post_ln.weight"), self.weights.get("audio.post_ln.bias")) {
            x = candle_nn::ops::layer_norm(&x, post_ln_w, post_ln_b, 1e-5)?;
        }

        // Audio Projector
        if let (Some(proj_w), Some(proj_b)) = (self.weights.get("projector.0.weight"), self.weights.get("projector.0.bias")) {
            x = matmul_3d_2d(&x, proj_w)?;
            x = x.broadcast_add(proj_b)?;
            x = x.gelu()?;
        }
        if let (Some(proj_w), Some(proj_b)) = (self.weights.get("projector.2.weight"), self.weights.get("projector.2.bias")) {
            x = matmul_3d_2d(&x, proj_w)?;
            x = x.broadcast_add(proj_b)?;
        }

        Ok(x)
    }

    /// Gemma-4 style Conformer encoder. `valid_mel_frames`: how many of
    /// `audio_values`' input mel frames are real audio vs. zero-padding
    /// (see `load_audio`) - propagated through both SSCP conv-subsampling
    /// stages and into the Conformer blocks' attention/lconv masking.
    fn encode_conformer(&self, audio_values: &Tensor, valid_mel_frames: usize) -> Result<Tensor> {
        let device = audio_values.device();
        // Dynamically resolve the working dtype from the weights
        let first_w = self.weights.values().next()
            .ok_or_else(|| anyhow!("No weights loaded in AudioEncoder"))?;
        let dtype = first_w.dtype();
        let audio_values = audio_values.to_dtype(dtype)?;
        let batch_size = audio_values.dim(0)?;

        // 1. SubSampleConvProjection
        // `audio_values` is (batch, freq=128, time) - unsqueeze(1) alone
        // would give (batch, channels=1, freq, time), i.e. H=freq, W=time.
        // But the reference's own SSCPConvBlock operates on
        // `x: [B, T, F, C]` (MLX channel-last, H=time, W=freq - confirmed
        // by its docstrings and by `mlx.nn.Conv2d`'s [B,H,W,C] convention),
        // and its conv WEIGHT is `[C_out, kH, kW, C_in]` with kH indexing a
        // TIME offset and kW indexing a FREQ offset. This project's GGUF-
        // loaded weight `a.conv1d.{0,1}.weight` has the SAME per-index
        // values (confirmed by direct comparison against the real
        // `mlx-community/gemma-4-e2b-it-4bit` weights:
        // `mine[c_out,c_in,kh,kw] == reference[c_out,kh,kw,c_in]` exactly),
        // meaning this kernel's kh axis is *semantically* a time offset and
        // kw a freq offset, regardless of which tensor axis order it's
        // stored in. A 3x3 conv kernel is not symmetric under swapping its
        // own two spatial axes, so convolving it against an (H=freq,W=time)
        // input - as this function did until this fix - applies the
        // kernel's time-offset weights to freq offsets and vice versa: a
        // real, silent, reference-confirmed bug (same weights, same input
        // data, plausible-looking but numerically wrong output - "right
        // ballpark, not matching per-position" was the exact symptom).
        // Transposing here to (batch, channels=1, time, freq) makes
        // everything downstream (masking on dim 2, the final
        // reshape-to-sequence) align with the reference's own (B,T,F,C)
        // convention with no further axis-juggling needed.
        let x = audio_values.unsqueeze(1)?.transpose(2, 3)?.contiguous()?;

        // Zero out invalid (padded) time steps BEFORE the conv, matching the
        // real Gemma-4 `SSCPConvBlock.__call__` (`x = mx.where(mask, 0.0,
        // x)`) - ported from mlx-vlm's actual `mlx_vlm/models/gemma4/
        // audio.py` (the real target architecture for this project's
        // "gemma-4" GGUF checkpoints - confirmed by running the real
        // `mlx-community/gemma-4-e2b-it-4bit` model on this machine and
        // reading its source directly, NOT the unrelated "Gemma 3n" model
        // in HF `transformers` an earlier pass in this session mistakenly
        // used as the reference). `x` is (batch, channels, time, freq) -
        // time is dim 2.
        let x = zero_invalid_time_steps(&x, 2, valid_mel_frames)?;

        // layer 0: Conv2d(stride=2, SYMMETRIC padding=1 on both time and
        // freq), LayerNorm (plain, channel-dim, no bias), ReLU - matching
        // `SSCPConvBlock`'s real `self.padding = (1,1,1,1)` and
        // `nn.LayerNorm(out_channels, ..., bias=False)`. An earlier pass
        // this session replaced this with an asymmetric "reverse-causal"
        // pad and a from-scratch `CumulativeGroupNorm`, both based on the
        // wrong ("Gemma 3n") reference - reverted now that the real
        // architecture is confirmed to use neither.
        let conv0_w = self.weights.get("a.conv1d.0.weight")
            .ok_or_else(|| anyhow!("a.conv1d.0.weight not found"))?;
        let norm0_w = self.weights.get("a.conv1d.0.norm.weight")
            .ok_or_else(|| anyhow!("a.conv1d.0.norm.weight not found"))?;

        let x = x.conv2d(conv0_w, 1, 2, 1, 1)?; // padding 1, stride 2

        // Permute to (batch, time, freq, channels) for LayerNorm
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;

        let valid_len0 = conv_stride2_out_len(valid_mel_frames);
        let norm0_b = Tensor::zeros(norm0_w.dim(0)?, dtype, device)?;
        let x = candle_nn::ops::layer_norm(&x, norm0_w, &norm0_b, 1e-6)?;
        let x = x.relu()?;
        let x = x.permute((0, 3, 1, 2))?; // Back to (batch, channels, h=time, w=freq)
        let x = zero_invalid_time_steps(&x, 2, valid_len0)?;

        // layer 1: same symmetric padding, plain LayerNorm, ReLU
        let conv1_w = self.weights.get("a.conv1d.1.weight")
            .ok_or_else(|| anyhow!("a.conv1d.1.weight not found"))?;
        let norm1_w = self.weights.get("a.conv1d.1.norm.weight")
            .ok_or_else(|| anyhow!("a.conv1d.1.norm.weight not found"))?;

        let x = x.contiguous()?.conv2d(conv1_w, 1, 2, 1, 1)?;
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let valid_len1 = conv_stride2_out_len(valid_len0);
        let norm1_b = Tensor::zeros(norm1_w.dim(0)?, dtype, device)?;
        let x = candle_nn::ops::layer_norm(&x, norm1_w, &norm1_b, 1e-6)?;
        let x = x.relu()?;
        let x = x.permute((0, 3, 1, 2))?; // (batch, channels=32, time=750, freq=32)
        let x = zero_invalid_time_steps(&x, 2, valid_len1)?;

        // Reshape to sequence: (batch, seq_len, hidden_dim). `x` is
        // (batch, channels, time, freq) here - permuting to
        // (batch, time, freq, channels) makes the reshape a pure "merge the
        // last two dims" no-op, exactly matching the reference's own
        // `B,T,F,C = x.shape; x = x.reshape(B,T,F*C)`.
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let seq_len = x.dim(1)?;
        let x = x.reshape((batch_size, seq_len, self.hidden_dim))?;

        // input_proj_linear
        let ip_w = self.weights.get("a.input_projection.weight")
            .ok_or_else(|| anyhow!("a.input_projection.weight not found"))?;
        let mut x = matmul_3d_2d(&x, ip_w)?;

        // 2. Position Embeddings
        // Gemma4AudioRelPositionalEncoding
        // See the comment on the identical constant in the attention forward pass
        // below: this is a fixed architectural constant, not GGUF-metadata-exposable.
        let chunk_size = 12;
        let rel_len = chunk_size + 1;

        let pos_ids: Vec<f32> = (0..rel_len).map(|v| (rel_len - 1 - v) as f32).collect();
        let position_ids = Tensor::from_vec(pos_ids, (rel_len, 1), device)?.to_dtype(dtype)?;

        let num_timescales = self.hidden_dim / 2;
        let min_timescale = 1.0f32;
        let max_timescale = 10000.0f32;
        let log_timescale_increment = (max_timescale / min_timescale).ln() / ((num_timescales - 1) as f32).max(1.0);
        let timescales: Vec<f32> = (0..num_timescales)
            .map(|i| min_timescale * (-(i as f32) * log_timescale_increment).exp())
            .collect();
        let inv_timescales = Tensor::from_vec(timescales, (1, 1, num_timescales), device)?.to_dtype(dtype)?;

        let scaled_time = position_ids.unsqueeze(0)?.broadcast_mul(&inv_timescales)?;
        let sin_t = scaled_time.sin()?;
        let cos_t = scaled_time.cos()?;
        let pos_embed = Tensor::cat(&[sin_t, cos_t], 2)?; // shape [1, rel_len, hidden_dim]

        // 3. Conformer Blocks (12 layers)
        for i in 0..self.num_layers {
            x = self.forward_conformer_block(i, &x, &pos_embed, rel_len, valid_len1)?;
        }

        // 4. Output Projector (output_proj)
        let op_w = self.weights.get("a.pre_encode.out.weight")
            .ok_or_else(|| anyhow!("a.pre_encode.out.weight not found"))?;
        let op_b = self.weights.get("a.pre_encode.out.bias")
            .ok_or_else(|| anyhow!("a.pre_encode.out.bias not found"))?;

        let x = matmul_3d_2d(&x, op_w)?;
        let x = x.broadcast_add(op_b)?;

        // Reference's final step (`AudioEncoder.__call__`): force EXACTLY
        // zero for any position at/past the valid length, regardless of
        // any small nonzero "leakage" the lconv1d's causal receptive field
        // may have re-introduced near the valid/invalid boundary inside the
        // last block (each block re-masks before its own lconv1d, but
        // nothing re-masks AFTER the last block's lconv1d until this final
        // pass). Without this, those few leaked-nonzero frames (plus
        // whatever splice_audio_embeddings does with the full fixed-size
        // 750-frame output regardless of real clip length) get spliced into
        // the LLM's input embeddings as if they were real audio content.
        zero_invalid_time_steps(&x, 1, valid_len1)
    }

    fn forward_conformer_block(&self, i: usize, x: &Tensor, pos_embed: &Tensor, rel_len: usize, valid_seq_len: usize) -> Result<Tensor> {
        let _device = x.device();
        let dtype = x.dtype();
        let grad_clip = if dtype == DType::F16 { 65504.0f32 } else { 1e10f32 };

        // 1. feed_forward1
        let ffw1_in = x.clamp(-grad_clip, grad_clip)?;
        let ffw1_norm = self.weights.get(&format!("a.blk.{}.ffn_norm.weight", i))
            .ok_or_else(|| anyhow!("ffn_norm for layer {} not found", i))?;
        let ffw1_normed = rms_norm(&ffw1_in, Some(ffw1_norm), 1e-6)?;

        let ffw1_w1 = self.weights.get(&format!("a.blk.{}.ffn_up.weight", i))
            .ok_or_else(|| anyhow!("ffn_up weight for layer {} not found", i))?;
        let ffw1_w1_imin = self.weights.get(&format!("a.blk.{}.ffn_up.input_min", i));
        let ffw1_w1_imax = self.weights.get(&format!("a.blk.{}.ffn_up.input_max", i));
        let ffw1_w1_omin = self.weights.get(&format!("a.blk.{}.ffn_up.output_min", i));
        let ffw1_w1_omax = self.weights.get(&format!("a.blk.{}.ffn_up.output_max", i));

        let ffw1_w2 = self.weights.get(&format!("a.blk.{}.ffn_down.weight", i))
            .ok_or_else(|| anyhow!("ffn_down weight for layer {} not found", i))?;
        let ffw1_w2_imin = self.weights.get(&format!("a.blk.{}.ffn_down.input_min", i));
        let ffw1_w2_imax = self.weights.get(&format!("a.blk.{}.ffn_down.input_max", i));
        let ffw1_w2_omin = self.weights.get(&format!("a.blk.{}.ffn_down.output_min", i));
        let ffw1_w2_omax = self.weights.get(&format!("a.blk.{}.ffn_down.output_max", i));

        let mut f1 = clippable_linear_forward(&ffw1_normed, ffw1_w1, ffw1_w1_imin, ffw1_w1_imax, ffw1_w1_omin, ffw1_w1_omax)?;
        f1 = f1.silu()?;
        let f1 = clippable_linear_forward(&f1, ffw1_w2, ffw1_w2_imin, ffw1_w2_imax, ffw1_w2_omin, ffw1_w2_omax)?;

        let f1_post_norm = self.weights.get(&format!("a.blk.{}.ffn_post_norm.weight", i))
            .ok_or_else(|| anyhow!("ffn_post_norm weight for layer {} not found", i))?;
        let f1_post_normed = rms_norm(&f1.clamp(-grad_clip, grad_clip)?, Some(f1_post_norm), 1e-6)?;

        // Add scaled residual: residual_weight = 0.5
        let x = x.add(&(f1_post_normed * 0.5)?)?;

        // 2. self_attn
        let residual = x.clone();
        let attn_in = x.clamp(-grad_clip, grad_clip)?;
        let attn_norm1 = self.weights.get(&format!("a.blk.{}.attn_pre_norm.weight", i))
            .ok_or_else(|| anyhow!("attn_pre_norm weight for layer {} not found", i))?;
        let attn_norm1_out = rms_norm(&attn_in, Some(attn_norm1), 1e-6)?;

        // q, k, v projections
        let q_w = self.weights.get(&format!("a.blk.{}.attn_q.weight", i))
            .ok_or_else(|| anyhow!("attn_q weight for layer {} not found", i))?;
        let q_imin = self.weights.get(&format!("a.blk.{}.attn_q.input_min", i));
        let q_imax = self.weights.get(&format!("a.blk.{}.attn_q.input_max", i));
        let q_omin = self.weights.get(&format!("a.blk.{}.attn_q.output_min", i));
        let q_omax = self.weights.get(&format!("a.blk.{}.attn_q.output_max", i));

        let k_w = self.weights.get(&format!("a.blk.{}.attn_k.weight", i))
            .ok_or_else(|| anyhow!("attn_k weight for layer {} not found", i))?;
        let k_imin = self.weights.get(&format!("a.blk.{}.attn_k.input_min", i));
        let k_imax = self.weights.get(&format!("a.blk.{}.attn_k.input_max", i));
        let k_omin = self.weights.get(&format!("a.blk.{}.attn_k.output_min", i));
        let k_omax = self.weights.get(&format!("a.blk.{}.attn_k.output_max", i));

        let v_w = self.weights.get(&format!("a.blk.{}.attn_v.weight", i))
            .ok_or_else(|| anyhow!("attn_v weight for layer {} not found", i))?;
        let v_imin = self.weights.get(&format!("a.blk.{}.attn_v.input_min", i));
        let v_imax = self.weights.get(&format!("a.blk.{}.attn_v.input_max", i));
        let v_omin = self.weights.get(&format!("a.blk.{}.attn_v.output_min", i));
        let v_omax = self.weights.get(&format!("a.blk.{}.attn_v.output_max", i));

        let q = clippable_linear_forward(&attn_norm1_out, q_w, q_imin, q_imax, q_omin, q_omax)?;
        let k = clippable_linear_forward(&attn_norm1_out, k_w, k_imin, k_imax, k_omin, k_omax)?;
        let v = clippable_linear_forward(&attn_norm1_out, v_w, v_imin, v_imax, v_omin, v_omax)?;

        // Chunked local attention
        let (batch_size, seq_len, hidden_size) = q.dims3()?;
        let num_heads = self.num_heads;
        let head_dim = self.hidden_dim / self.num_heads;
        // These are NOT exposed as GGUF metadata keys anywhere in llama.cpp's/HF's
        // Gemma-4-Conformer export (unlike `hidden_dim`/`num_layers` above, which
        // come from `clip.audio.embedding_length`/`clip.audio.block_count`) — they
        // are fixed architectural constants of this specific Conformer variant's
        // local/chunked attention window, not a per-checkpoint hyperparameter.
        // Intentionally hardcoded; not a model-agnosticism bug.
        let chunk_size = 12;
        let max_past_horizon = 12;
        let max_future_horizon = 0;
        let context_size = 24;

        // q = q * q_scale * softplus(per_dim_scale)
        let per_dim_scale = self.weights.get(&format!("a.blk.{}.per_dim_scale.weight", i))
            .ok_or_else(|| anyhow!("per_dim_scale weight for layer {} not found", i))?;
        let q_scale = (head_dim as f64).powf(-0.5) / 2.0f64.ln();
        let q_scale_factor = (softplus(per_dim_scale)? * q_scale)?;
        let q_reshaped = q.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let q_scaled = q_reshaped.broadcast_mul(&q_scale_factor)?;

        // Fixed K-side multiplier `ln(1+e)/ln(2) ~= 1.894` (`softplus(1.0)/ln(2)`).
        // An earlier pass this session removed this, reasoning (from the
        // unrelated "Gemma 3n" HF `transformers` reference) that no such
        // K-scale exists in this architecture family - that reference was
        // simply the wrong model. Confirmed by reading the REAL Gemma-4
        // reference directly (`mlx_vlm/models/gemma4/audio.py`,
        // `self.k_scale = math.log(1 + math.e) / math.log(2)`, applied as
        // `k = k * self.k_scale` - i.e. exactly this constant): restored.
        let k_scale = (1.0f64 + std::f64::consts::E).ln() / 2.0f64.ln();
        let k_reshaped = k.reshape((batch_size, seq_len, num_heads, head_dim))?;
        let k_scaled = (k_reshaped * k_scale)?;
        let v_reshaped = v.reshape((batch_size, seq_len, num_heads, head_dim))?;

        // convert to block/context
        let num_blocks = (seq_len + chunk_size - 1) / chunk_size;
        let pad = num_blocks * chunk_size - seq_len;
        let pad_tensor = Tensor::zeros((batch_size, pad, num_heads, head_dim), q.dtype(), q.device())?;
        let q_padded = Tensor::cat(&[q_scaled, pad_tensor.clone()], 1)?.contiguous()?;
        let query_states = q_padded.reshape((batch_size, num_blocks, chunk_size, num_heads, head_dim))?;

        let key_states = extract_block_context(&k_scaled, max_past_horizon, max_future_horizon, chunk_size, context_size)?;
        let value_states = extract_block_context(&v_reshaped, max_past_horizon, max_future_horizon, chunk_size, context_size)?;

        // relative_key_states = relative_k_proj(pos_embed)
        let rel_k_w = self.weights.get(&format!("a.blk.{}.attn_k_rel.weight", i))
            .ok_or_else(|| anyhow!("attn_k_rel weight for layer {} not found", i))?;
        let relative_key_states = matmul_3d_2d(pos_embed, rel_k_w)?;
        let relative_key_states = relative_key_states.reshape((rel_len, num_heads, head_dim))?;

        let queries = query_states.permute((0, 3, 1, 2, 4))?.contiguous()?;
        let keys = key_states.permute((0, 3, 1, 4, 2))?.contiguous()?;

        let b_sz = batch_size * num_heads * num_blocks;
        let queries_3d = queries.reshape((b_sz, chunk_size, head_dim))?;
        let keys_3d = keys.reshape((b_sz, head_dim, context_size))?;
        let matrix_ac = queries_3d.matmul(&keys_3d)?
            .reshape((batch_size, num_heads, num_blocks, chunk_size, context_size))?;

        let queries_flat = queries.contiguous()?.reshape((batch_size, num_heads, num_blocks * chunk_size, head_dim))?;
        let rel_keys = relative_key_states.permute((1, 2, 0))?.unsqueeze(0)?.contiguous()?;
        let matrix_bd = queries_flat.matmul(&rel_keys)?;
        let matrix_bd = matrix_bd.reshape((batch_size, num_heads, num_blocks, chunk_size, rel_len))?;
        let matrix_bd = rel_shift(&matrix_bd, context_size, chunk_size)?;

        let attn_weights = (matrix_ac + matrix_bd)?;
        let softcap = 50.0f64;
        let attn_weights = (attn_weights / softcap)?;
        let attn_weights = attn_weights.tanh()?;
        let attn_weights = (attn_weights * softcap)?;

        // Combined local-causal + validity mask, ported from the reference's
        // `create_local_causal_valid_mask` (tril-based construction, worked
        // out algebraically and cross-checked term-by-term against the
        // actual tril/diagonal semantics) plus its per-block validity check
        // against `audio_mel_mask`. This is exactly the masking real
        // inference engines apply for variable-length audio (llama.cpp's
        // Gemma-4 Conformer support processes the same fixed 30s-buffer
        // convention; vLLM's variable-length audio attention excludes
        // padded positions from attention the same way) - without it, every
        // block attends across its full context window as if the entire
        // fixed-size buffer were real audio, including any trailing
        // zero-padding from clips shorter than the buffer.
        let mask_bias = conformer_attention_mask_bias(
            num_blocks, chunk_size, context_size,
            max_past_horizon, max_future_horizon, valid_seq_len,
            attn_weights.device(),
        )?;
        let mask_bias = mask_bias.to_dtype(attn_weights.dtype())?;
        let attn_weights = attn_weights.broadcast_add(&mask_bias)?;

        let attn_probs = candle_nn::ops::softmax(&attn_weights, 4)?;

        let value_states_perm = value_states.permute((0, 3, 1, 2, 4))?.contiguous()?;
        let attn_probs_3d = attn_probs.reshape((b_sz, chunk_size, context_size))?;
        let value_states_perm_3d = value_states_perm.reshape((b_sz, context_size, head_dim))?;
        let attn_output = attn_probs_3d.matmul(&value_states_perm_3d)?
            .reshape((batch_size, num_heads, num_blocks, chunk_size, head_dim))?;

        let attn_output = attn_output.permute((0, 2, 3, 1, 4))?.contiguous()?;
        let attn_output = attn_output.reshape((batch_size, num_blocks * chunk_size, hidden_size))?;
        let attn_output = attn_output.narrow(1, 0, seq_len)?;

        let post_w = self.weights.get(&format!("a.blk.{}.attn_out.weight", i))
            .ok_or_else(|| anyhow!("attn_out weight for layer {} not found", i))?;
        let post_imin = self.weights.get(&format!("a.blk.{}.attn_out.input_min", i));
        let post_imax = self.weights.get(&format!("a.blk.{}.attn_out.input_max", i));
        let post_omin = self.weights.get(&format!("a.blk.{}.attn_out.output_min", i));
        let post_omax = self.weights.get(&format!("a.blk.{}.attn_out.output_max", i));
        let attn_output = clippable_linear_forward(&attn_output, post_w, post_imin, post_imax, post_omin, post_omax)?;

        // Reference clamps the self_attn output before norm_post_attn
        // (`x = mx.clip(x, -grad_clip, grad_clip); x = residual + norm_post_attn(x)`
        // in `ConformerBlock.__call__`) - this clamp was missing here.
        let attn_output = attn_output.clamp(-grad_clip, grad_clip)?;
        let attn_norm2 = self.weights.get(&format!("a.blk.{}.attn_post_norm.weight", i))
            .ok_or_else(|| anyhow!("attn_post_norm weight for layer {} not found", i))?;
        let attn_norm2_out = rms_norm(&attn_output, Some(attn_norm2), 1e-6)?;

        let x = residual.add(&attn_norm2_out)?;

        // Zero out padded positions before the light-conv, matching the
        // reference's `validity_mask_for_lconv` - otherwise the depthwise
        // conv's receptive field would blend real content with whatever
        // (already other-wise-masked-to-near-zero, but not exactly zero
        // once past a RMSNorm's rescaling) values sit in the padded tail.
        let x = if valid_seq_len < seq_len {
            let validity: Vec<f32> = (0..seq_len).map(|t| if t < valid_seq_len { 1.0 } else { 0.0 }).collect();
            let validity = Tensor::from_vec(validity, (1, seq_len, 1), x.device())?.to_dtype(x.dtype())?;
            x.broadcast_mul(&validity)?
        } else {
            x
        };

        // 3. lconv1d
        let residual = x.clone();
        let conv_in = x.clamp(-grad_clip, grad_clip)?;
        let conv_norm1 = self.weights.get(&format!("a.blk.{}.norm_conv.weight", i))
            .ok_or_else(|| anyhow!("norm_conv weight for layer {} not found", i))?;
        let conv_norm1_out = rms_norm(&conv_in, Some(conv_norm1), 1e-6)?;

        let lconv_pw1_w = self.weights.get(&format!("a.blk.{}.conv_pw1.weight", i))
            .ok_or_else(|| anyhow!("conv_pw1 weight for layer {} not found", i))?;
        let lconv_pw1_imin = self.weights.get(&format!("a.blk.{}.conv_pw1.input_min", i));
        let lconv_pw1_imax = self.weights.get(&format!("a.blk.{}.conv_pw1.input_max", i));
        let lconv_pw1_omin = self.weights.get(&format!("a.blk.{}.conv_pw1.output_min", i));
        let lconv_pw1_omax = self.weights.get(&format!("a.blk.{}.conv_pw1.output_max", i));
        let conv_proj1 = clippable_linear_forward(&conv_norm1_out, lconv_pw1_w, lconv_pw1_imin, lconv_pw1_imax, lconv_pw1_omin, lconv_pw1_omax)?;

        let conv_glu = glu(&conv_proj1)?;

        // Depthwise Conv1d
        let conv_dw_w = self.weights.get(&format!("a.blk.{}.conv_dw.weight", i))
            .ok_or_else(|| anyhow!("conv_dw weight for layer {} not found", i))?;
        let kernel_size = conv_dw_w.dim(conv_dw_w.rank() - 1)?;
        let conv_dw_w_reshaped = conv_dw_w.reshape((hidden_size, 1, kernel_size))?;

        let x_trans = conv_glu.transpose(1, 2)?.contiguous()?;
        let pad_len = kernel_size - 1;
        let pad_left_t = Tensor::zeros((batch_size, hidden_size, pad_len), x_trans.dtype(), x_trans.device())?;
        let x_padded = Tensor::cat(&[pad_left_t, x_trans], 2)?.contiguous()?;
        let conv_dw_out = x_padded.conv1d(&conv_dw_w_reshaped, 0, 1, 1, hidden_size)?;
        let conv_dw_out = conv_dw_out.transpose(1, 2)?;

        let conv_dw_clamped = conv_dw_out.clamp(-grad_clip, grad_clip)?;
        let conv_norm2 = self.weights.get(&format!("a.blk.{}.conv_norm.weight", i))
            .ok_or_else(|| anyhow!("conv_norm weight for layer {} not found", i))?;
        let conv_norm2_out = rms_norm(&conv_dw_clamped, Some(conv_norm2), 1e-6)?;
        let conv_norm2_act = conv_norm2_out.silu()?;

        let lconv_pw2_w = self.weights.get(&format!("a.blk.{}.conv_pw2.weight", i))
            .ok_or_else(|| anyhow!("conv_pw2 weight for layer {} not found", i))?;
        let lconv_pw2_imin = self.weights.get(&format!("a.blk.{}.conv_pw2.input_min", i));
        let lconv_pw2_imax = self.weights.get(&format!("a.blk.{}.conv_pw2.input_max", i));
        let lconv_pw2_omin = self.weights.get(&format!("a.blk.{}.conv_pw2.output_min", i));
        let lconv_pw2_omax = self.weights.get(&format!("a.blk.{}.conv_pw2.output_max", i));
        let conv_proj2 = clippable_linear_forward(&conv_norm2_act, lconv_pw2_w, lconv_pw2_imin, lconv_pw2_imax, lconv_pw2_omin, lconv_pw2_omax)?;

        let x = residual.add(&conv_proj2)?;

        // 4. feed_forward2
        let ffw2_in = x.clamp(-grad_clip, grad_clip)?;
        let ffw2_norm = self.weights.get(&format!("a.blk.{}.ffn_norm_1.weight", i))
            .ok_or_else(|| anyhow!("ffn_norm_1 weight for layer {} not found", i))?;
        let ffw2_normed = rms_norm(&ffw2_in, Some(ffw2_norm), 1e-6)?;

        let ffw2_w1 = self.weights.get(&format!("a.blk.{}.ffn_up_1.weight", i))
            .ok_or_else(|| anyhow!("ffn_up_1 weight for layer {} not found", i))?;
        let ffw2_w1_imin = self.weights.get(&format!("a.blk.{}.ffn_up_1.input_min", i));
        let ffw2_w1_imax = self.weights.get(&format!("a.blk.{}.ffn_up_1.input_max", i));
        let ffw2_w1_omin = self.weights.get(&format!("a.blk.{}.ffn_up_1.output_min", i));
        let ffw2_w1_omax = self.weights.get(&format!("a.blk.{}.ffn_up_1.output_max", i));

        let ffw2_w2 = self.weights.get(&format!("a.blk.{}.ffn_down_1.weight", i))
            .ok_or_else(|| anyhow!("ffn_down_1 weight for layer {} not found", i))?;
        let ffw2_w2_imin = self.weights.get(&format!("a.blk.{}.ffn_down_1.input_min", i));
        let ffw2_w2_imax = self.weights.get(&format!("a.blk.{}.ffn_down_1.input_max", i));
        let ffw2_w2_omin = self.weights.get(&format!("a.blk.{}.ffn_down_1.output_min", i));
        let ffw2_w2_omax = self.weights.get(&format!("a.blk.{}.ffn_down_1.output_max", i));

        let mut f2 = clippable_linear_forward(&ffw2_normed, ffw2_w1, ffw2_w1_imin, ffw2_w1_imax, ffw2_w1_omin, ffw2_w1_omax)?;
        f2 = f2.silu()?;
        let f2 = clippable_linear_forward(&f2, ffw2_w2, ffw2_w2_imin, ffw2_w2_imax, ffw2_w2_omin, ffw2_w2_omax)?;

        let f2_post_norm = self.weights.get(&format!("a.blk.{}.ffn_post_norm_1.weight", i))
            .ok_or_else(|| anyhow!("ffn_post_norm_1 weight for layer {} not found", i))?;
        let f2_post_normed = rms_norm(&f2.clamp(-grad_clip, grad_clip)?, Some(f2_post_norm), 1e-6)?;

        let x = x.add(&(f2_post_normed * 0.5)?)?;

        // 5. norm_out
        let out_in = x.clamp(-grad_clip, grad_clip)?;
        let out_norm = self.weights.get(&format!("a.blk.{}.norm_out.weight", i));
        let x = rms_norm(&out_in, out_norm, 1e-6)?;

        Ok(x)
    }
}

/// Additive attention-logit bias (`0.0` where a query/key pair is allowed to
/// attend, large-negative where it isn't), combining two conditions exactly
/// as the reference's `Gemma3nAudioAttention.create_local_causal_valid_mask`
/// + its per-block validity check against `audio_mel_mask` do:
///
///   - **Local causal window**: query row `w` (0..chunk_size) may attend to
///     context column `c` (0..context_size) iff `c >= w` (causal: never
///     attend to a strictly-future position within the block) AND
///     `c <= w + max_past_horizon + max_future_horizon` (bounded window).
///     Derived algebraically from the reference's `tril`/`diagonal`
///     construction and cross-checked term-by-term against `tril`'s actual
///     semantics (not assumed) - see the module's audit notes.
///   - **Validity**: context column `c` in block `b` corresponds to
///     absolute key position `b*chunk_size - max_past_horizon + c` (the
///     same left-padding convention `extract_block_context` itself uses);
///     that position is valid iff it falls within `[0, valid_seq_len)`.
///
/// Returns a `[1, 1, num_blocks, chunk_size, context_size]` tensor,
/// broadcastable against attention logits shaped
/// `[batch, num_heads, num_blocks, chunk_size, context_size]`.
#[allow(clippy::too_many_arguments)]
fn conformer_attention_mask_bias(
    num_blocks: usize,
    chunk_size: usize,
    context_size: usize,
    max_past_horizon: usize,
    max_future_horizon: usize,
    valid_seq_len: usize,
    device: &Device,
) -> Result<Tensor> {
    // Large enough to zero out a masked position's softmax probability, but
    // safely representable in F16 (max ~65504) without relying on
    // overflow-to-infinity rounding during the dtype cast below.
    const MASKED_BIAS: f32 = -1.0e4;
    let mut data = vec![0.0f32; num_blocks * chunk_size * context_size];
    for b in 0..num_blocks {
        for w in 0..chunk_size {
            for c in 0..context_size {
                let causal_ok = c >= w && c <= w + max_past_horizon + max_future_horizon;
                let abs_key_pos = b as i64 * chunk_size as i64 - max_past_horizon as i64 + c as i64;
                let valid_ok = abs_key_pos >= 0 && (abs_key_pos as usize) < valid_seq_len;
                if !(causal_ok && valid_ok) {
                    let idx = (b * chunk_size + w) * context_size + c;
                    data[idx] = MASKED_BIAS;
                }
            }
        }
    }
    let t = Tensor::from_vec(data, (1, 1, num_blocks, chunk_size, context_size), device)?;
    Ok(t)
}

/// Output time-axis length of one of the SSCP stage's stride-2 conv2d
/// calls, given `input_len` real (valid) time steps. Matches the actual
/// conv arithmetic `(input_len + total_pad - kernel) / stride + 1` with
/// `total_pad=2, kernel=3, stride=2`, which simplifies to `(input_len-1)/2+1`
/// - and is exactly equivalent to how the real Gemma-4 implementation
/// downsamples its own boolean validity mask by simple striding
/// (`mask[:, ::2]`; `ceil(V/2) == (V-1)//2+1` for any `V >= 1`). Used to
/// track how many of the SSCP output's time positions are still "real"
/// after each downsampling stage.
fn conv_stride2_out_len(input_len: usize) -> usize {
    if input_len == 0 { 0 } else { (input_len - 1) / 2 + 1 }
}

/// Zeroes out time steps at or past `valid_len` along dimension `time_dim`,
/// leaving earlier steps untouched. Ported from the real Gemma-4
/// `SSCPConvBlock`'s masking (`x = mx.where(mask, 0.0, x)`, confirmed by
/// running `mlx-community/gemma-4-e2b-it-4bit` on this machine and reading
/// `mlx_vlm/models/gemma4/audio.py` directly) - excludes zero-padded
/// buffer positions from the conv/norm computation rather than processing
/// them as if they were real audio.
fn zero_invalid_time_steps(x: &Tensor, time_dim: usize, valid_len: usize) -> Result<Tensor> {
    let t = x.dim(time_dim)?;
    if valid_len >= t {
        return Ok(x.clone());
    }
    let mut shape = vec![1usize; x.rank()];
    shape[time_dim] = t;
    let vals: Vec<f32> = (0..t).map(|i| if i < valid_len { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::from_vec(vals, shape, x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_mul(&mask)?)
}

fn rms_norm(x: &Tensor, weight: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let original_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean_sq = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let x_normed = x_f32.broadcast_div(&(mean_sq + eps)?.sqrt()?)?;
    let res = if let Some(w) = weight {
        let w_f32 = w.to_dtype(DType::F32)?;
        x_normed.broadcast_mul(&w_f32)?
    } else {
        x_normed
    };
    Ok(res.to_dtype(original_dtype)?)
}

fn to_f32_scalar(t: &Tensor) -> Result<f32> {
    let t = t.to_dtype(DType::F32)?.flatten_all()?;
    let vec = t.to_vec1::<f32>()?;
    if vec.is_empty() {
        anyhow::bail!("Empty tensor for scalar conversion");
    }
    Ok(vec[0])
}

fn clippable_linear_forward(
    x: &Tensor,
    weight: &Tensor,
    input_min: Option<&Tensor>,
    input_max: Option<&Tensor>,
    output_min: Option<&Tensor>,
    output_max: Option<&Tensor>,
) -> Result<Tensor> {
    let mut hidden_states = x.clone();

    if let (Some(imin), Some(imax)) = (input_min, input_max) {
        let min_val = to_f32_scalar(imin)?;
        let max_val = to_f32_scalar(imax)?;
        hidden_states = hidden_states.clamp(min_val, max_val)?;
    }

    hidden_states = matmul_3d_2d(&hidden_states, weight)?;

    if let (Some(omin), Some(omax)) = (output_min, output_max) {
        let min_val = to_f32_scalar(omin)?;
        let max_val = to_f32_scalar(omax)?;
        hidden_states = hidden_states.clamp(min_val, max_val)?;
    }

    Ok(hidden_states)
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    Ok((x.exp()? + 1.0)?.log()?)
}

fn glu(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(2)?;
    let half = last_dim / 2;
    let a = x.narrow(2, 0, half)?;
    let b = x.narrow(2, half, half)?;
    Ok(a.broadcast_mul(&candle_nn::ops::sigmoid(&b)?)?)
}

/// Ported from the reference's `AudioAttention._extract_block_context`:
/// left-pad by `max_past_horizon`, right-pad by `max_future_horizon +
/// chunk_size - 1`, then slice a `context_size`-wide window per block
/// starting at `b * chunk_size` in the *padded* tensor.
///
/// The previous version of this function padded on the right ONLY (by the
/// query-side amount `num_blocks*chunk_size - seq_len`) and derived each
/// block's start via `(block_start + chunk_size).saturating_sub(context_size)`
/// - which happens to equal the correct real-position start
/// `b*chunk_size - max_past_horizon` for every block except **block 0**,
/// where the true start is negative (`-max_past_horizon`) and should be
/// realized as `max_past_horizon` zero-padded positions followed by real
/// data. `saturating_sub` instead clamped it to `0`, silently shifting
/// block 0's entire context window `max_past_horizon` positions to the
/// right - a real, reference-confirmed bug (caught by the leftover
/// underscore-prefixed unused `_max_past_horizon`/`_max_future_horizon`
/// parameters, which should have been the tell that padding wasn't actually
/// using them). Confirmed by reading `_extract_block_context` in
/// `mlx_vlm/models/gemma4/audio.py` directly, which explicitly
/// left-pads before indexing.
fn extract_block_context(
    x: &Tensor,
    max_past_horizon: usize,
    max_future_horizon: usize,
    chunk_size: usize,
    context_size: usize,
) -> Result<Tensor> {
    let (batch_size, _seq_len, num_heads, head_dim) = x.dims4()?;
    let pad_left = max_past_horizon;
    let pad_right = max_future_horizon + chunk_size - 1;
    let left_pad = Tensor::zeros((batch_size, pad_left, num_heads, head_dim), x.dtype(), x.device())?;
    let right_pad = Tensor::zeros((batch_size, pad_right, num_heads, head_dim), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[left_pad, x.clone(), right_pad], 1)?.contiguous()?;
    let t_padded = x_padded.dim(1)?;
    let num_blocks = (t_padded - context_size) / chunk_size + 1;

    let mut blocks = Vec::new();
    for b in 0..num_blocks {
        let start = b * chunk_size;
        let slice = x_padded.narrow(1, start, context_size)?;
        blocks.push(slice);
    }
    Ok(Tensor::stack(&blocks, 1)?)
}

fn rel_shift(x: &Tensor, context_size: usize, _chunk_size: usize) -> Result<Tensor> {
    let (batch_size, num_heads, num_blocks, block_size, position_length) = x.dims5()?;
    let pad_len = context_size + 1 - position_length;
    let pad_tensor = Tensor::zeros((batch_size, num_heads, num_blocks, block_size, pad_len), x.dtype(), x.device())?;
    let x = Tensor::cat(&[x.clone(), pad_tensor], 4)?.contiguous()?;
    let x = x.reshape((batch_size, num_heads, num_blocks, block_size * (context_size + 1)))?;
    let x = x.narrow(3, 0, block_size * context_size)?;
    Ok(x.reshape((batch_size, num_heads, num_blocks, block_size, context_size))?)
}

/// Map the several real-world Whisper-derived tensor-naming conventions
/// (`self_attn.{q,k,v}_proj`, `self_attn.out_proj`/`self_attn.dense`,
/// `mlp.fc1`/`fc2` or `dense_h_to_4h`/`dense_4h_to_h`, etc.) onto the
/// canonical `audio.*` names `encode_whisper` expects, and fuses the
/// separate q/k/v projections into one `attn_qkv` weight per layer.
fn normalize_audio_tensors(
    raw: HashMap<String, Tensor>,
    num_layers: usize,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut normalized = HashMap::new();
    let mut q_weights = HashMap::new();
    let mut k_weights = HashMap::new();
    let mut v_weights = HashMap::new();
    let mut q_biases = HashMap::new();
    let mut k_biases = HashMap::new();
    let mut v_biases = HashMap::new();

    for (k, v) in raw {
        if k.contains("audio_encoder.conv1.weight") {
            normalized.insert("audio.conv1.weight".to_string(), v);
        } else if k.contains("audio_encoder.conv1.bias") {
            normalized.insert("audio.conv1.bias".to_string(), v);
        } else if k.contains("audio_encoder.conv2.weight") {
            normalized.insert("audio.conv2.weight".to_string(), v);
        } else if k.contains("audio_encoder.conv2.bias") {
            normalized.insert("audio.conv2.bias".to_string(), v);
        } else if k.contains("audio_encoder.positional_embedding") {
            normalized.insert("audio.pos_embed.weight".to_string(), v);
        } else if k.contains("audio_encoder.ln_post.weight") {
            normalized.insert("audio.post_ln.weight".to_string(), v);
        } else if k.contains("audio_encoder.ln_post.bias") {
            normalized.insert("audio.post_ln.bias".to_string(), v);
        } else if k.contains("audio_projector.linear_1.weight") || k.contains("audio_projector.0.weight") {
            normalized.insert("projector.0.weight".to_string(), v);
        } else if k.contains("audio_projector.linear_1.bias") || k.contains("audio_projector.0.bias") {
            normalized.insert("projector.0.bias".to_string(), v);
        } else if k.contains("audio_projector.linear_2.weight") || k.contains("audio_projector.2.weight") {
            normalized.insert("projector.2.weight".to_string(), v);
        } else if k.contains("audio_projector.linear_2.bias") || k.contains("audio_projector.2.bias") {
            normalized.insert("projector.2.bias".to_string(), v);
        } else if k.contains("audio_encoder.layers.") {
            let parts: Vec<&str> = k.split('.').collect();
            if let Some(idx_str) = parts.iter().position(|&p| p == "layers").and_then(|pos| parts.get(pos + 1)) {
                if let Ok(layer_idx) = idx_str.parse::<usize>() {
                    if k.contains("layer_norm1.weight") {
                        normalized.insert(format!("audio.layers.{}.ln1.weight", layer_idx), v);
                    } else if k.contains("layer_norm1.bias") {
                        normalized.insert(format!("audio.layers.{}.ln1.bias", layer_idx), v);
                    } else if k.contains("layer_norm2.weight") {
                        normalized.insert(format!("audio.layers.{}.ln2.weight", layer_idx), v);
                    } else if k.contains("layer_norm2.bias") {
                        normalized.insert(format!("audio.layers.{}.ln2.bias", layer_idx), v);
                    } else if k.contains("self_attn.out_proj.weight") || k.contains("self_attn.dense.weight") {
                        normalized.insert(format!("audio.layers.{}.attn_out.weight", layer_idx), v);
                    } else if k.contains("self_attn.out_proj.bias") || k.contains("self_attn.dense.bias") {
                        normalized.insert(format!("audio.layers.{}.attn_out.bias", layer_idx), v);
                    } else if k.contains("mlp.fc1.weight") || k.contains("mlp.dense_h_to_4h.weight") {
                        normalized.insert(format!("audio.layers.{}.ffn_up.weight", layer_idx), v);
                    } else if k.contains("mlp.fc1.bias") || k.contains("mlp.dense_h_to_4h.bias") {
                        normalized.insert(format!("audio.layers.{}.ffn_up.bias", layer_idx), v);
                    } else if k.contains("mlp.fc2.weight") || k.contains("mlp.dense_4h_to_h.weight") {
                        normalized.insert(format!("audio.layers.{}.ffn_down.weight", layer_idx), v);
                    } else if k.contains("mlp.fc2.bias") || k.contains("mlp.dense_4h_to_h.bias") {
                        normalized.insert(format!("audio.layers.{}.ffn_down.bias", layer_idx), v);
                    } else if k.contains("self_attn.q_proj.weight") {
                        q_weights.insert(layer_idx, v);
                    } else if k.contains("self_attn.k_proj.weight") {
                        k_weights.insert(layer_idx, v);
                    } else if k.contains("self_attn.v_proj.weight") {
                        v_weights.insert(layer_idx, v);
                    } else if k.contains("self_attn.q_proj.bias") {
                        q_biases.insert(layer_idx, v);
                    } else if k.contains("self_attn.k_proj.bias") {
                        k_biases.insert(layer_idx, v);
                    } else if k.contains("self_attn.v_proj.bias") {
                        v_biases.insert(layer_idx, v);
                    }
                }
            }
        }
    }

    for layer_idx in 0..num_layers {
        let q_w = q_weights.remove(&layer_idx);
        let k_w = k_weights.remove(&layer_idx);
        let v_w = v_weights.remove(&layer_idx);
        // Remember each projection's actual output dimension (from its weight)
        // before the weights are consumed below, so a missing bias for one of
        // Q/K/V can be zero-filled at the CORRECT shape instead of a bogus
        // length-1 placeholder that would break `broadcast_add` downstream.
        let q_out_dim = q_w.as_ref().map(|t| t.dim(0)).transpose()?;
        let k_out_dim = k_w.as_ref().map(|t| t.dim(0)).transpose()?;
        let v_out_dim = v_w.as_ref().map(|t| t.dim(0)).transpose()?;
        if let (Some(q), Some(k), Some(v)) = (&q_w, &k_w, &v_w) {
            let qkv = Tensor::cat(&[q, k, v], 0)?;
            normalized.insert(format!("audio.layers.{}.attn_qkv.weight", layer_idx), qkv);
        }
        let q_b = q_biases.remove(&layer_idx);
        let k_b = k_biases.remove(&layer_idx);
        let v_b = v_biases.remove(&layer_idx);
        if q_b.is_some() || k_b.is_some() || v_b.is_some() {
            let dtype = q_b.as_ref().or(k_b.as_ref()).or(v_b.as_ref())
                .map(|t| t.dtype())
                .ok_or_else(|| anyhow!("no q/k/v bias available for layer {}", layer_idx))?;
            let q_b = q_b.map(Ok).unwrap_or_else(|| Tensor::zeros(q_out_dim.unwrap_or(1), dtype, device))?;
            let k_b = k_b.map(Ok).unwrap_or_else(|| Tensor::zeros(k_out_dim.unwrap_or(1), dtype, device))?;
            let v_b = v_b.map(Ok).unwrap_or_else(|| Tensor::zeros(v_out_dim.unwrap_or(1), dtype, device))?;
            let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;
            normalized.insert(format!("audio.layers.{}.attn_qkv.bias", layer_idx), qkv_b);
        } else if let Some(qkv_w) = normalized.get(&format!("audio.layers.{}.attn_qkv.weight", layer_idx)) {
            let out_dim = qkv_w.dim(0)?;
            let zeros = Tensor::zeros(out_dim, qkv_w.dtype(), device)?;
            normalized.insert(format!("audio.layers.{}.attn_qkv.bias", layer_idx), zeros);
        }
    }

    Ok(normalized)
}

/// Returns `(mel_spectrogram, valid_frames)` - `valid_frames` is how many of
/// the tensor's time-axis positions are real audio vs. zero-padding (see
/// `gemma3n_num_frames`).
pub fn load_audio(path: &Path, device: &Device, architecture: AudioArchitecture) -> Result<(Tensor, usize)> {
    tracing::info!("Loading audio from {:?}", path);

    let src = std::fs::File::open(path)
        .context("Failed to open audio file")?;

    let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = symphonia::core::probe::Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &Default::default(), &Default::default())
        .context("Unsupported audio format")?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No supported audio track found"))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .context("Unsupported audio codec")?;

    let track_id = track.id;
    let mut pcm_data = Vec::new();
    let mut source_rate: Option<u32> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(symphonia::core::errors::Error::IoError(err)) if err.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(err) => return Err(err.into()),
        };

        if packet.track_id() != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(audio_buf) => {
                let spec = *audio_buf.spec();
                source_rate.get_or_insert(spec.rate);
                let mut sample_buf = symphonia::core::audio::AudioBuffer::<f32>::new(audio_buf.capacity() as u64, spec);
                audio_buf.convert(&mut sample_buf);

                let planes = sample_buf.planes();
                let num_frames = sample_buf.frames();
                let num_channels = planes.planes().len();

                if num_channels > 0 {
                    for i in 0..num_frames {
                        let mut sum = 0.0;
                        for ch in 0..num_channels {
                            sum += planes.planes()[ch][i];
                        }
                        pcm_data.push(sum / num_channels as f32);
                    }
                }
            }
            Err(symphonia::core::errors::Error::DecodeError(_)) => {
                continue;
            }
            Err(err) => return Err(err.into()),
        }
    }

    tracing::info!("Decoded {} raw mono audio samples", pcm_data.len());

    // The mel-spectrogram below assumes 16kHz input (25ms/400-sample frames,
    // 10ms/160-sample hop). Previously the source sample rate was decoded
    // and then silently discarded, so any non-16kHz file (the vast majority
    // of real-world audio: 44.1kHz/48kHz are far more common than 16kHz) was
    // fed through as-is, time/pitch-distorting it by ~2.75-3x before even
    // reaching feature extraction. Resample via linear interpolation
    // (adequate for feeding a mel-filterbank; not audiophile-grade, but a
    // real correction rather than silently ignoring the mismatch).
    if let Some(rate) = source_rate {
        if rate != 16000 && rate > 0 && !pcm_data.is_empty() {
            let ratio = 16000.0 / rate as f64;
            let resampled_len = ((pcm_data.len() as f64) * ratio).round() as usize;
            let mut resampled = Vec::with_capacity(resampled_len);
            for i in 0..resampled_len {
                let src_pos = i as f64 / ratio;
                let idx0 = src_pos.floor() as usize;
                let frac = (src_pos - idx0 as f64) as f32;
                let s0 = pcm_data.get(idx0).copied().unwrap_or(0.0);
                let s1 = pcm_data.get(idx0 + 1).copied().unwrap_or(s0);
                resampled.push(s0 + (s1 - s0) * frac);
            }
            tracing::info!(
                "Resampled audio from {} Hz to 16000 Hz ({} -> {} samples)",
                rate, pcm_data.len(), resampled.len()
            );
            pcm_data = resampled;
        }
    }

    let target_samples = 480000;
    let real_sample_count = pcm_data.len().min(target_samples);
    if pcm_data.len() < target_samples {
        pcm_data.resize(target_samples, 0.0);
    } else {
        pcm_data.truncate(target_samples);
    }

    match architecture {
        // Whisper was trained expecting a fixed, always-fully-padded 30s
        // buffer processed uniformly (no reference-confirmed masking
        // behavior for this path, unlike Gemma-4 below) - `3000` here
        // means "every frame is valid," a no-op for the caller's masking.
        AudioArchitecture::Whisper => Ok((whisper_mel_spectrogram(&pcm_data, device)?, 3000)),
        AudioArchitecture::GemmaConformer => {
            // real_sample_count is left-padded by FRAME_LENGTH/2 the same
            // way the mel computation itself is - see `gemma4_num_frames`.
            let padded_real_samples = real_sample_count + 320 / 2;
            let mel = gemma4_mel_spectrogram(&pcm_data, device)?;
            let valid_frames = gemma4_num_frames(padded_real_samples);
            Ok((mel, valid_frames))
        }
    }
}

/// Number of real (non-padded) mel frames Gemma-4's feature extractor
/// produces for `num_samples` of (semicausally left-padded) real audio -
/// same `_unfold` arithmetic as `gemma4_mel_spectrogram` itself
/// (`(len - (frame_length+1)) / hop + 1`). Used to build a validity mask
/// for the encoder (see `encode_conformer`) - real inference engines
/// (mlx-vlm's real Gemma-4 Conformer support, vLLM's variable-length
/// audio attention) all exclude padded positions from attention/
/// normalization rather than process them as if they were real audio.
fn gemma4_num_frames(num_samples: usize) -> usize {
    const FRAME_LENGTH: usize = 320;
    const HOP_LENGTH: usize = 160;
    let frame_size_for_unfold = FRAME_LENGTH + 1;
    if num_samples >= frame_size_for_unfold {
        (num_samples - frame_size_for_unfold) / HOP_LENGTH + 1
    } else {
        0
    }
}

/// Whisper's feature-extraction convention: 16kHz audio, 25ms/400-sample
/// Hann-windowed frames, 10ms/160-sample hop, 3000 frames for 30s of audio,
/// power spectrum, and Whisper's specific final normalization (clip dynamic
/// range to the top 8 natural-log units, then rescale to roughly [-1, 1]).
/// This replaces a previous placeholder that computed only a per-frame
/// scalar RMS energy fanned out via a fixed sine envelope — a real but
/// content-blind "spectrogram" carrying no actual frequency information.
fn whisper_mel_spectrogram(pcm_data: &[f32], device: &Device) -> Result<Tensor> {
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;
    const SAMPLE_RATE: f32 = 16000.0;
    const NUM_MEL_BINS: usize = 80;
    let window = hann_window(N_FFT);
    let mel_filters = build_mel_filterbank(NUM_MEL_BINS, N_FFT, SAMPLE_RATE, 0.0, SAMPLE_RATE / 2.0);

    let mut mel_data = vec![0.0f32; NUM_MEL_BINS * 3000];
    let mut frame_buf = vec![0.0f32; N_FFT];
    for frame in 0..3000 {
        let start_idx = frame * HOP_LENGTH;
        for (i, w) in window.iter().enumerate() {
            let idx = start_idx + i;
            let sample = pcm_data.get(idx).copied().unwrap_or(0.0);
            frame_buf[i] = sample * w;
        }
        let power_spectrum = dft_power_spectrum(&frame_buf);
        for (bin, filter) in mel_filters.iter().enumerate() {
            let energy: f32 = filter.iter().zip(power_spectrum.iter()).map(|(f, p)| f * p).sum();
            mel_data[bin * 3000 + frame] = energy.max(1e-10).ln();
        }
    }

    // Match Whisper's final normalization: clip the dynamic range to the top
    // 8 natural-log units, then rescale to roughly [-1, 1] so the values fall
    // in the range the audio encoders were actually trained on.
    let log_max = mel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    for v in mel_data.iter_mut() {
        *v = v.max(log_max - 8.0);
        *v = (*v + 4.0) / 4.0;
    }

    let t = Tensor::from_vec(mel_data, (1, NUM_MEL_BINS, 3000), device)?;
    Ok(t)
}

/// Gemma-4's real audio feature extractor (`Gemma4AudioFeatureExtractor`),
/// ported exactly from `mlx-vlm`'s reference source
/// (`mlx_vlm/models/gemma4/audio_feature_extractor.py`, read directly
/// after confirming it against the actual `mlx-community/gemma-4-e2b-
/// it-4bit` checkpoint's `processor_config.json`, and validated by running
/// that real model on this machine with a real speech sample) — an
/// EARLIER pass this session had implemented "Gemma 3n"'s (HF
/// `transformers`) feature extractor instead, a different, unrelated
/// model with a materially different front-end; every constant below is
/// now confirmed against the real target architecture:
///   - 20ms/320-sample frames (not 32ms/512), 10ms/160-sample hop (same).
///   - FFT length 512 (not 1024 - no "FFT overdrive" doubling for this model).
///   - Mel filterbank spans 0-8000 Hz (not 125-7600 Hz).
///   - NO preemphasis (coefficient 0.0 - the opposite correction from the
///     "Gemma 3n" pass, which had added HTK preemphasis that doesn't
///     belong here at all).
///   - Semicausal left-padding: `frame_length/2` (160) zero samples
///     prepended before framing, so the first frame is centered at t=0 -
///     entirely missing from the previous implementation.
///   - `ln(mel + 1e-3)` (additive floor, not `ln(max(mel, floor))`).
///   - Magnitude spectrum (`|STFT|`), not power spectrum (`|STFT|^2`) -
///     this part was already correct from the "Gemma 3n" pass.
fn gemma4_mel_spectrogram(pcm_data: &[f32], device: &Device) -> Result<Tensor> {
    const SAMPLE_RATE: f32 = 16000.0;
    const FRAME_LENGTH: usize = 320; // round(16000 * 20ms / 1000)
    const HOP_LENGTH: usize = 160; // round(16000 * 10ms / 1000)
    const FFT_LENGTH: usize = 512; // 2^ceil(log2(320)), no fft_overdrive for this model
    const NUM_MEL_BINS: usize = 128;
    const MEL_MIN_FREQ: f32 = 0.0;
    const MEL_MAX_FREQ: f32 = 8000.0;
    const MEL_FLOOR: f32 = 1e-3;

    // Semicausal left-padding: the real extractor prepends frame_length/2
    // zero samples so the first frame is centered at t=0, matching the
    // real model's own `_extract_spectrogram`.
    let pad_left = FRAME_LENGTH / 2;
    let mut padded = vec![0.0f32; pad_left + pcm_data.len()];
    padded[pad_left..].copy_from_slice(pcm_data);

    let window = hann_window(FRAME_LENGTH);
    let mel_filters = build_mel_filterbank(NUM_MEL_BINS, FFT_LENGTH, SAMPLE_RATE, MEL_MIN_FREQ, MEL_MAX_FREQ);

    // `_unfold(waveform, size=frame_length+1, step=hop_length)`: the extra
    // leading sample is a leftover of the (disabled, for this model)
    // preemphasis codepath, but the frame count arithmetic still uses it.
    let frame_size_for_unfold = FRAME_LENGTH + 1;
    let num_frames = if padded.len() >= frame_size_for_unfold {
        (padded.len() - frame_size_for_unfold) / HOP_LENGTH + 1
    } else {
        0
    };

    let mut mel_data = vec![0.0f32; NUM_MEL_BINS * num_frames];
    let mut fft_input = vec![0.0f32; FFT_LENGTH];

    for frame in 0..num_frames {
        let start_idx = frame * HOP_LENGTH;
        // No preemphasis for this model (coefficient 0.0): frame is simply
        // `frames_to_process[..., :-1]` - the first FRAME_LENGTH samples of
        // the unfolded (frame_length+1)-sample window.
        for (i, w) in window.iter().enumerate() {
            let sample = padded.get(start_idx + i).copied().unwrap_or(0.0);
            fft_input[i] = sample * w;
        }
        for slot in fft_input.iter_mut().skip(FRAME_LENGTH) {
            *slot = 0.0;
        }

        let power_spectrum = dft_power_spectrum(&fft_input);
        for (bin, filter) in mel_filters.iter().enumerate() {
            let magnitude_energy: f32 = filter.iter().zip(power_spectrum.iter())
                .map(|(f, p)| f * p.sqrt())
                .sum();
            mel_data[bin * num_frames + frame] = (magnitude_energy + MEL_FLOOR).ln();
        }
    }

    let t = Tensor::from_vec(mel_data, (1, NUM_MEL_BINS, num_frames), device)?;
    Ok(t)
}

fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.contiguous()?.matmul(&rhs.t()?.unsqueeze(0)?.contiguous()?)?)
}

/// Periodic Hann window (`w[n] = 0.5 - 0.5*cos(2*pi*n/n_len)`), matching
/// both the real Gemma-4 feature extractor's explicit formula
/// (`mlx_vlm/models/gemma4/audio_feature_extractor.py`: "Periodic Hann
/// window... Matches HuggingFace Transformers (signal.hann_window with
/// periodic=True)") and PyTorch/Whisper's own default
/// (`torch.hann_window(n, periodic=True)`, the default). The previous
/// version divided by `n-1` (the *symmetric* Hann window) instead of `n` -
/// a real, reference-confirmed bug: it produced a mel-spectrogram "close
/// but not bit-exact" to ground truth, with the discrepancy compounding
/// through 12 Conformer blocks into a much larger encoder-output error.
fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

/// Triangular mel filterbank (HTK mel scale), one row per mel bin, each row
/// spanning the `n_fft/2 + 1` real-valued power-spectrum bins. Used to
/// collapse a linear-frequency power spectrum into a perceptually-scaled
/// log-mel spectrogram, matching standard Whisper/wav2vec2-style audio
/// feature extraction.
fn build_mel_filterbank(num_mel_bins: usize, n_fft: usize, sample_rate: f32, f_min: f32, f_max: f32) -> Vec<Vec<f32>> {
    let num_fft_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> = (0..num_mel_bins + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();
    // Hz -> FFT-bin-index: each bin k represents frequency k*sample_rate/n_fft,
    // so the inverse is `Hz * n_fft / sample_rate` - confirmed against real
    // reference mel-spectrogram values (`mlx_vlm`'s `_mel_filter_bank`,
    // `all_freqs = arange(n_freq_bins) * (sample_rate / (2*(n_freq_bins-1)))`,
    // which is algebraically the same `n_fft` scaling since
    // `2*(n_fft/2+1-1) = n_fft`). A stray `+1` here previously introduced a
    // small but real, numerically-confirmed discrepancy against the real
    // Gemma-4 model's own mel output (bit-exact on a pure-silence frame,
    // consistently off by ~1-2% per bin on a frame with real signal).
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| n_fft as f32 * mel_to_hz(m) / sample_rate)
        .collect();

    let mut filters = vec![vec![0.0f32; num_fft_bins]; num_mel_bins];
    for (m, filter) in filters.iter_mut().enumerate() {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];
        for (k, slot) in filter.iter_mut().enumerate() {
            let kf = k as f32;
            if kf >= left && kf <= center && center > left {
                *slot = (kf - left) / (center - left);
            } else if kf > center && kf <= right && right > center {
                *slot = (right - kf) / (right - center);
            }
        }
    }
    filters
}

/// Power spectrum (|FFT|^2) of a single already-windowed frame, for the
/// `n/2 + 1` non-redundant real-input frequency bins. A direct O(n^2) DFT
/// rather than a radix-2 FFT: `frame.len()` (400, Whisper's standard 25ms
/// window at 16kHz) isn't a power of 2, and this runs once per ~10ms hop at
/// audio-load time, not in the per-token decode hot path, so the simplicity
/// of a direct DFT is worth more here than the extra complexity (and new
/// dependency) a general-length FFT would need.
fn dft_power_spectrum(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let num_bins = n / 2 + 1;
    let mut power = vec![0.0f32; num_bins];
    for (k, slot) in power.iter_mut().enumerate() {
        let angle_step = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        for (i, &sample) in frame.iter().enumerate() {
            let angle = angle_step * i as f32;
            re += sample * angle.cos();
            im += sample * angle.sin();
        }
        *slot = re * re + im * im;
    }
    power
}

#[cfg(test)]
mod arch_detection_tests {
    use super::*;

    fn dummy_tensor() -> Tensor {
        Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap()
    }

    #[test]
    fn detects_whisper_from_audio_encoder_prefix() {
        let mut weights = HashMap::new();
        weights.insert("audio_encoder.layers.0.self_attn.q_proj.weight".to_string(), dummy_tensor());
        assert_eq!(detect_architecture(&weights), AudioArchitecture::Whisper);
    }

    #[test]
    fn detects_whisper_from_projector_prefix() {
        let mut weights = HashMap::new();
        weights.insert("audio_projector.linear_1.weight".to_string(), dummy_tensor());
        assert_eq!(detect_architecture(&weights), AudioArchitecture::Whisper);
    }

    #[test]
    fn defaults_to_gemma_conformer_for_native_gguf_names() {
        let mut weights = HashMap::new();
        weights.insert("a.conv1d.0.weight".to_string(), dummy_tensor());
        weights.insert("a.blk.0.attn_q.weight".to_string(), dummy_tensor());
        assert_eq!(detect_architecture(&weights), AudioArchitecture::GemmaConformer);
    }

    #[test]
    fn mel_bins_differ_by_architecture() {
        assert_eq!(AudioArchitecture::Whisper.num_mel_bins(), 80);
        assert_eq!(AudioArchitecture::GemmaConformer.num_mel_bins(), 128);
    }

    /// Proves the mel-spectrogram is real spectral analysis, not the old
    /// placeholder (a per-frame scalar energy fanned out via a fixed sine
    /// envelope, which is architecture/content-blind): a 1kHz tone and a
    /// 4kHz tone at the same amplitude must peak in genuinely different mel
    /// bins. The old placeholder could never satisfy this — its shape across
    /// bins was a fixed `sin(bin/n * pi)` curve independent of input
    /// frequency, identical for every distinct pure tone.
    #[test]
    fn mel_filterbank_distinguishes_different_frequencies() {
        let n_fft = 400;
        let sample_rate = 16000.0f32;
        let num_mel_bins = 80;
        let window = hann_window(n_fft);
        let filters = build_mel_filterbank(num_mel_bins, n_fft, sample_rate, 0.0, sample_rate / 2.0);

        let tone_bin = |freq_hz: f32| -> usize {
            let frame: Vec<f32> = (0..n_fft)
                .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate).sin() * window[i])
                .collect();
            let power = dft_power_spectrum(&frame);
            filters
                .iter()
                .enumerate()
                .map(|(bin, filter)| {
                    let energy: f32 = filter.iter().zip(power.iter()).map(|(f, p)| f * p).sum();
                    (bin, energy)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(bin, _)| bin)
                .unwrap()
        };

        let low_bin = tone_bin(1000.0);
        let high_bin = tone_bin(4000.0);
        assert!(
            high_bin > low_bin,
            "a 4kHz tone must peak in a higher mel bin than a 1kHz tone (got {low_bin} vs {high_bin})"
        );
    }

    #[test]
    fn mel_filterbank_rows_are_nonzero() {
        let filters = build_mel_filterbank(80, 400, 16000.0, 0.0, 8000.0);
        for (i, row) in filters.iter().enumerate() {
            assert!(row.iter().any(|&v| v > 0.0), "mel filter row {i} is all zero");
        }
    }
}
