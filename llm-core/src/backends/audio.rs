use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, Context};
use candle_core::{Tensor, Device, DType};
use symphonia::core::audio::Signal;

pub struct AudioEncoder {
    weights: HashMap<String, Tensor>,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub projection_dim: usize,
}

impl AudioEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!("Loading audio encoder from {:?}", path);

        let mut raw_weights = HashMap::new();
        let mut hidden_dim = 1024;
        let mut num_layers = 12;
        let mut num_heads = 8;
        let mut projection_dim = 1024;

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

            if let Some(v) = get_metadata_u32("clip.audio.embedding_length") { hidden_dim = v as usize; }
            if let Some(v) = get_metadata_u32("clip.audio.block_count") { num_layers = v as usize; }
            if let Some(v) = get_metadata_u32("clip.audio.attention.head_count") { num_heads = v as usize; }
            if let Some(v) = get_metadata_u32("clip.audio.projection_dim") { projection_dim = v as usize; }

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

        Ok(Self {
            weights: raw_weights,
            hidden_dim,
            num_layers,
            num_heads,
            projection_dim,
        })
    }

    pub fn encode(&self, audio_values: &Tensor) -> Result<Tensor> {
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

pub fn load_audio(path: &Path, device: &Device) -> Result<Tensor> {
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

    let target_samples = 480000;
    if pcm_data.len() < target_samples {
        pcm_data.resize(target_samples, 0.0);
    } else {
        pcm_data.truncate(target_samples);
    }

    let num_mel_bins = 128;
    let mut mel_data = vec![0.0f32; num_mel_bins * 3000];
    for frame in 0..3000 {
        let start_idx = frame * 160;
        let mut power = 0.0f32;
        let window_len = 320;
        for offset in 0..window_len {
            if start_idx + offset < pcm_data.len() {
                let sample = pcm_data[start_idx + offset];
                power += sample * sample;
            }
        }
        power = (power / window_len as f32).sqrt();

        for bin in 0..num_mel_bins {
            let factor = ((bin as f32 / num_mel_bins as f32) * std::f32::consts::PI).sin();
            mel_data[bin * 3000 + frame] = power * factor;
        }
    }

    let t = Tensor::from_vec(mel_data, (1, num_mel_bins, 3000), device)?;
    Ok(t)
}

fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.contiguous()?.matmul(&rhs.t()?.unsqueeze(0)?.contiguous()?)?)
}
