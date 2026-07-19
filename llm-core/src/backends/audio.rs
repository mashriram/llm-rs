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

    pub fn encode(&self, audio_values: &Tensor) -> Result<Tensor> {
        match self.architecture {
            AudioArchitecture::GemmaConformer => self.encode_conformer(audio_values),
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

    /// Gemma-4 style Conformer encoder.
    fn encode_conformer(&self, audio_values: &Tensor) -> Result<Tensor> {
        let device = audio_values.device();
        // Dynamically resolve the working dtype from the weights
        let first_w = self.weights.values().next()
            .ok_or_else(|| anyhow!("No weights loaded in AudioEncoder"))?;
        let dtype = first_w.dtype();
        let audio_values = audio_values.to_dtype(dtype)?;
        let batch_size = audio_values.dim(0)?;

        // 1. SubSampleConvProjection
        // Unsqueeze to (batch, 1, 128, 3000)
        let x = audio_values.unsqueeze(1)?.contiguous()?;

        // layer 0: Conv2d(stride=2, padding=1), LayerNorm, ReLU
        let conv0_w = self.weights.get("a.conv1d.0.weight")
            .ok_or_else(|| anyhow!("a.conv1d.0.weight not found"))?;
        let norm0_w = self.weights.get("a.conv1d.0.norm.weight")
            .ok_or_else(|| anyhow!("a.conv1d.0.norm.weight not found"))?;

        let x = x.conv2d(conv0_w, 1, 2, 1, 1)?; // padding 1, stride 2

        // Permute to (batch, h, w, channels) for LayerNorm
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;

        let norm0_b = Tensor::zeros(128, dtype, device)?;
        let x = candle_nn::ops::layer_norm(&x, norm0_w, &norm0_b, 1e-5)?;
        let x = x.relu()?;
        let x = x.permute((0, 3, 1, 2))?; // Back to (batch, channels, h, w)

        // layer 1: Conv2d(stride=2, padding=1), LayerNorm, ReLU
        let conv1_w = self.weights.get("a.conv1d.1.weight")
            .ok_or_else(|| anyhow!("a.conv1d.1.weight not found"))?;
        let norm1_w = self.weights.get("a.conv1d.1.norm.weight")
            .ok_or_else(|| anyhow!("a.conv1d.1.norm.weight not found"))?;

        let x = x.contiguous()?.conv2d(conv1_w, 1, 2, 1, 1)?; // padding 1, stride 2
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let norm1_b = Tensor::zeros(32, dtype, device)?;
        let x = candle_nn::ops::layer_norm(&x, norm1_w, &norm1_b, 1e-5)?;
        let x = x.relu()?;
        let x = x.permute((0, 3, 1, 2))?; // (batch, 32, 32, 750)

        // Reshape to sequence: (batch, seq_len, hidden_dim)
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        let seq_len = x.dim(2)?;
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
            x = self.forward_conformer_block(i, &x, &pos_embed, rel_len)?;
        }

        // 4. Output Projector (output_proj)
        let op_w = self.weights.get("a.pre_encode.out.weight")
            .ok_or_else(|| anyhow!("a.pre_encode.out.weight not found"))?;
        let op_b = self.weights.get("a.pre_encode.out.bias")
            .ok_or_else(|| anyhow!("a.pre_encode.out.bias not found"))?;

        let x = matmul_3d_2d(&x, op_w)?;
        let x = x.broadcast_add(op_b)?;

        Ok(x)
    }

    fn forward_conformer_block(&self, i: usize, x: &Tensor, pos_embed: &Tensor, rel_len: usize) -> Result<Tensor> {
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

        let attn_norm2 = self.weights.get(&format!("a.blk.{}.attn_post_norm.weight", i))
            .ok_or_else(|| anyhow!("attn_post_norm weight for layer {} not found", i))?;
        let attn_norm2_out = rms_norm(&attn_output, Some(attn_norm2), 1e-6)?;

        let x = residual.add(&attn_norm2_out)?;

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

fn extract_block_context(
    x: &Tensor,
    _max_past_horizon: usize,
    _max_future_horizon: usize,
    chunk_size: usize,
    context_size: usize,
) -> Result<Tensor> {
    let (batch_size, seq_len, num_heads, head_dim) = x.dims4()?;
    let num_blocks = (seq_len + chunk_size - 1) / chunk_size;
    let pad = num_blocks * chunk_size - seq_len;
    let pad_tensor = Tensor::zeros((batch_size, pad, num_heads, head_dim), x.dtype(), x.device())?;
    let x_padded = Tensor::cat(&[x.clone(), pad_tensor], 1)?.contiguous()?;

    let mut blocks = Vec::new();
    for b in 0..num_blocks {
        let block_start = b * chunk_size;
        let context_start = (block_start + chunk_size).saturating_sub(context_size);
        let context_len = context_size;
        let slice = x_padded.narrow(1, context_start, context_len)?;
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

pub fn load_audio(path: &Path, device: &Device, num_mel_bins: usize) -> Result<Tensor> {
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
    if pcm_data.len() < target_samples {
        pcm_data.resize(target_samples, 0.0);
    } else {
        pcm_data.truncate(target_samples);
    }

    // Real log-mel spectrogram (Whisper's feature-extraction convention:
    // 16kHz audio, 25ms/400-sample Hann-windowed frames, 10ms/160-sample hop,
    // 3000 frames for 30s of audio). This replaces a previous placeholder
    // that computed only a per-frame scalar RMS energy and fanned it out
    // across mel bins via a fixed sine envelope — a real but content-blind
    // "spectrogram" that could not carry any actual frequency information
    // regardless of what audio encoder architecture consumed it (Whisper or
    // Gemma-Conformer). Every audio-capable model was affected equally,
    // since the bug was here, not in any per-architecture encoder code.
    const N_FFT: usize = 400;
    const HOP_LENGTH: usize = 160;
    const SAMPLE_RATE: f32 = 16000.0;
    let window = hann_window(N_FFT);
    let mel_filters = build_mel_filterbank(num_mel_bins, N_FFT, SAMPLE_RATE);

    let mut mel_data = vec![0.0f32; num_mel_bins * 3000];
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

    let t = Tensor::from_vec(mel_data, (1, num_mel_bins, 3000), device)?;
    Ok(t)
}

fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.contiguous()?.matmul(&rhs.t()?.unsqueeze(0)?.contiguous()?)?)
}

fn hann_window(n: usize) -> Vec<f32> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos())
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
fn build_mel_filterbank(num_mel_bins: usize, n_fft: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let num_fft_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sample_rate / 2.0);
    let mel_points: Vec<f32> = (0..num_mel_bins + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| (n_fft as f32 + 1.0) * mel_to_hz(m) / sample_rate)
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
        let filters = build_mel_filterbank(num_mel_bins, n_fft, sample_rate);

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
        let filters = build_mel_filterbank(80, 400, 16000.0);
        for (i, row) in filters.iter().enumerate() {
            assert!(row.iter().any(|&v| v > 0.0), "mel filter row {i} is all zero");
        }
    }
}
