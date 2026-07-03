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
        let mut hidden_dim = 1280;
        let mut num_layers = 32;
        let mut num_heads = 20;
        let mut projection_dim = 2560;

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
            for name in model.tensor_infos.keys() {
                let qtensor = model.tensor(&mut file, name, &cpu)
                    .context(format!("Failed to load audio tensor {}", name))?;
                let tensor = qtensor.dequantize(&cpu)
                    .context(format!("Failed to dequantize audio tensor {}", name))?
                    .to_dtype(DType::F16)?
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

            for (k, v) in loaded {
                raw_weights.insert(k, v.to_dtype(DType::F16)?);
            }
        }

        let weights = normalize_audio_tensors(raw_weights, num_layers, device)?;

        Ok(Self {
            weights,
            hidden_dim,
            num_layers,
            num_heads,
            projection_dim,
        })
    }

    pub fn encode(&self, audio_values: &Tensor) -> Result<Tensor> {
        // 1. Conv1D Feature Extraction
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
        // From [batch, hidden_dim, seq_len] -> permute(0, 2, 1)
        x = x.transpose(1, 2)?;

        // 2. Positional Embeddings
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

        // 3. Transformer Encoder Blocks
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

        // 4. Post LN
        if let (Some(post_ln_w), Some(post_ln_b)) = (self.weights.get("audio.post_ln.weight"), self.weights.get("audio.post_ln.bias")) {
            x = candle_nn::ops::layer_norm(&x, post_ln_w, post_ln_b, 1e-5)?;
        }

        // 5. Audio Projector
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
}

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
        if k.contains("audio_encoder.conv1.weight") || k.contains("audio_encoder.conv1.weight") {
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
        if let (Some(q), Some(k), Some(v)) = (q_weights.remove(&layer_idx), k_weights.remove(&layer_idx), v_weights.remove(&layer_idx)) {
            let qkv = Tensor::cat(&[&q, &k, &v], 0)?;
            normalized.insert(format!("audio.layers.{}.attn_qkv.weight", layer_idx), qkv);
        }
        let q_b = q_biases.remove(&layer_idx);
        let k_b = k_biases.remove(&layer_idx);
        let v_b = v_biases.remove(&layer_idx);
        if q_b.is_some() || k_b.is_some() || v_b.is_some() {
            let q_b = q_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
            let k_b = k_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
            let v_b = v_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
            let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;
            normalized.insert(format!("audio.layers.{}.attn_qkv.bias", layer_idx), qkv_b);
        } else {
            if let Some(qkv_w) = normalized.get(&format!("audio.layers.{}.attn_qkv.weight", layer_idx)) {
                let out_dim = qkv_w.dim(0)?;
                let zeros = Tensor::zeros(out_dim, DType::F16, device)?;
                normalized.insert(format!("audio.layers.{}.attn_qkv.bias", layer_idx), zeros);
            }
        }
    }

    Ok(normalized)
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

    let mut mel_data = vec![0.0f32; 80 * 3000];
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

        for bin in 0..80 {
            let factor = ((bin as f32 / 80.0) * std::f32::consts::PI).sin();
            mel_data[bin * 3000 + frame] = power * factor;
        }
    }

    let t = Tensor::from_vec(mel_data, (1, 80, 3000), device)?;
    Ok(t)
}

fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.contiguous()?.matmul(&rhs.unsqueeze(0)?.contiguous()?)?)
}

