use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, Context};
use candle_core::{Tensor, Device, DType};

pub struct VisionEncoder {
    weights: HashMap<String, Tensor>,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub projection_dim: usize,
    pub spatial_merge_size: usize,
    pub is_deepstack_layers: Vec<bool>,
}

impl VisionEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        tracing::info!("Loading vision encoder mmproj / weights from {:?}", path);

        let mut raw_weights = HashMap::new();
        let mut image_size = 768;
        let mut patch_size = 16;
        let mut hidden_dim = 1024;
        let mut num_layers = 24;
        let mut num_heads = 16;
        let mut projection_dim = 2560;
        let mut spatial_merge_size = 2;
        let mut is_deepstack_layers = vec![false; 24];

        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        if is_gguf {
            let mut file = std::fs::File::open(path)
                .context(format!("Failed to open mmproj GGUF file: {:?}", path))?;
            let model = candle_core::quantized::gguf_file::Content::read(&mut file)
                .context("Failed to read GGUF mmproj content")?;

            let get_metadata_u32 = |key: &str| -> Option<u32> {
                match model.metadata.get(key) {
                    Some(candle_core::quantized::gguf_file::Value::U32(v)) => Some(*v),
                    Some(candle_core::quantized::gguf_file::Value::I32(v)) => Some(*v as u32),
                    _ => None,
                }
            };

            if let Some(v) = get_metadata_u32("clip.vision.image_size") { image_size = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.patch_size") { patch_size = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.embedding_length") { hidden_dim = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.block_count") { num_layers = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.attention.head_count") { num_heads = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.projection_dim") { projection_dim = v as usize; }
            if let Some(v) = get_metadata_u32("clip.vision.spatial_merge_size") { spatial_merge_size = v as usize; }

            is_deepstack_layers = match model.metadata.get("clip.vision.is_deepstack_layers") {
                Some(candle_core::quantized::gguf_file::Value::Array(arr)) => {
                    arr.iter().map(|v| match v {
                        candle_core::quantized::gguf_file::Value::Bool(b) => *b,
                        _ => false,
                    }).collect()
                }
                _ => vec![false; num_layers],
            };

            let cpu = Device::Cpu;
            for name in model.tensor_infos.keys() {
                let qtensor = model.tensor(&mut file, name, &cpu)
                    .context(format!("Failed to load mmproj tensor {}", name))?;
                let tensor = qtensor.dequantize(&cpu)
                    .context(format!("Failed to dequantize mmproj tensor {}", name))?
                    .to_dtype(DType::F16)?
                    .to_device(device)?;
                raw_weights.insert(name.clone(), tensor);
            }
        } else {
            // Load from Safetensors directory or file
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

            // If config.json is present in the parent or same folder, parse it
            let config_json = if path.is_dir() {
                Some(path.join("config.json"))
            } else if let Some(parent) = path.parent() {
                Some(parent.join("config.json"))
            } else {
                None
            };

            if let Some(ref conf_path) = config_json {
                if conf_path.exists() {
                    if let Ok(meta) = crate::model::config::parse_config(conf_path) {
                        image_size = meta.vision_image_size.unwrap_or(image_size);
                        patch_size = meta.vision_patch_size.unwrap_or(patch_size);
                        hidden_dim = meta.vision_hidden_dim.unwrap_or(hidden_dim);
                        num_layers = meta.vision_num_layers.unwrap_or(num_layers);
                        num_heads = meta.vision_num_heads.unwrap_or(num_heads);
                        projection_dim = meta.vision_projection_dim.unwrap_or(projection_dim);
                        spatial_merge_size = meta.spatial_merge_size.unwrap_or(spatial_merge_size);
                        is_deepstack_layers = meta.is_deepstack_layers.unwrap_or_else(|| vec![false; num_layers]);
                    }
                }
            }
        }

        let weights = normalize_vision_tensors(raw_weights, num_layers, device)?;

        Ok(Self {
            weights,
            image_size,
            patch_size,
            hidden_dim,
            num_layers,
            num_heads,
            projection_dim,
            spatial_merge_size,
            is_deepstack_layers,
        })
    }

    pub fn encode(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let _device = pixel_values.device();
        let _dtype = pixel_values.dtype();

        // 1. Patch Embedding (Conv2D: patch_embed.weight shape [hidden_dim, 3, patch_size, patch_size])
        let patch_emb_w = self.weights.get("vision.patch_embed.weight")
            .ok_or_else(|| anyhow!("vision.patch_embed.weight weight not found"))?;
        let patch_emb_b = self.weights.get("vision.patch_embed.bias");

        let mut x = pixel_values.conv2d(patch_emb_w, self.patch_size, 0, 1, 1)?;

        if let Some(bias) = patch_emb_b {
            let bias_reshaped = bias.reshape((1, self.hidden_dim, 1, 1))?;
            x = x.broadcast_add(&bias_reshaped)?;
        }

        // Reshape [1, hidden_dim, H_out, W_out] -> [1, hidden_dim, patches] -> [1, patches, hidden_dim]
        let (b, c, h, w) = x.dims4()?;
        let num_patches = h * w;
        x = x.reshape((b, c, num_patches))?.permute((0, 2, 1))?;

        // 2. Add Positional Embedding (pos_embed.weight shape [num_patches, hidden_dim])
        if let Some(pos_emb) = self.weights.get("vision.pos_embed.weight") {
            // Some models include a class token in pos_embed length (e.g. 2305 vs 2304)
            let pos_len = pos_emb.dim(0)?;
            let pos_to_add = if pos_len > num_patches {
                pos_emb.narrow(0, pos_len - num_patches, num_patches)?
            } else {
                pos_emb.clone()
            };
            x = x.broadcast_add(&pos_to_add.reshape((1, num_patches, c))?)?;
        }

        // Fuyu/Unified Bypass: if num_layers is 0, we bypass the trunk entirely and project directly!
        if self.num_layers == 0 {
            // Direct projection
            if let Some(mm_0_w) = self.weights.get("projector.0.weight") {
                let mm_0_b = self.weights.get("projector.0.bias");
                x = x.matmul(&mm_0_w.t()?)?;
                if let Some(b) = mm_0_b {
                    x = x.broadcast_add(b)?;
                }
            }
            return Ok(x);
        }

        // 3. ViT Block Layers
        let mut deepstack_outputs = Vec::new();
        let head_dim = self.hidden_dim / self.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        for i in 0..self.num_layers {
            // LayerNorm 1
            let ln1_w = self.weights.get(&format!("vision.layers.{}.ln1.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln1.weight not found", i))?;
            let ln1_b = self.weights.get(&format!("vision.layers.{}.ln1.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln1.bias not found", i))?;
            let x_ln1 = candle_nn::ops::layer_norm(&x, ln1_w, ln1_b, 1e-6)?;

            // Self Attention QKV
            let qkv_w = self.weights.get(&format!("vision.layers.{}.attn_qkv.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_qkv.weight not found", i))?;
            let qkv_b = self.weights.get(&format!("vision.layers.{}.attn_qkv.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_qkv.bias not found", i))?;

            let qkv = x_ln1.matmul(&qkv_w.t()?)?.broadcast_add(qkv_b)?;
            let qkv_chunks = qkv.chunk(3, 2)?;
            let q = qkv_chunks[0].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;
            let k = qkv_chunks[1].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;
            let v = qkv_chunks[2].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;

            let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            let attn_out = attn_probs.matmul(&v)?;

            let attn_out = attn_out.permute((0, 2, 1, 3))?.reshape((1, num_patches, self.hidden_dim))?;

            // Output projection
            let out_w = self.weights.get(&format!("vision.layers.{}.attn_out.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_out.weight not found", i))?;
            let out_b = self.weights.get(&format!("vision.layers.{}.attn_out.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_out.bias not found", i))?;
            let attn_out = attn_out.matmul(&out_w.t()?)?.broadcast_add(out_b)?;

            x = (x + attn_out)?;

            // LayerNorm 2
            let ln2_w = self.weights.get(&format!("vision.layers.{}.ln2.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln2.weight not found", i))?;
            let ln2_b = self.weights.get(&format!("vision.layers.{}.ln2.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln2.bias not found", i))?;
            let x_ln2 = candle_nn::ops::layer_norm(&x, ln2_w, ln2_b, 1e-6)?;

            // MLP
            let ffn_up_w = self.weights.get(&format!("vision.layers.{}.ffn_up.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_up.weight not found", i))?;
            let ffn_up_b = self.weights.get(&format!("vision.layers.{}.ffn_up.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_up.bias not found", i))?;
            let ffn_down_w = self.weights.get(&format!("vision.layers.{}.ffn_down.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_down.weight not found", i))?;
            let ffn_down_b = self.weights.get(&format!("vision.layers.{}.ffn_down.bias", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_down.bias not found", i))?;

            let mlp = x_ln2.matmul(&ffn_up_w.t()?)?.broadcast_add(ffn_up_b)?;
            let mlp = mlp.gelu()?;
            let mlp = mlp.matmul(&ffn_down_w.t()?)?.broadcast_add(ffn_down_b)?;

            x = (x + mlp)?;

            // DeepStack integration
            if self.is_deepstack_layers.get(i).copied().unwrap_or(false) {
                let merged = spatial_merge(&x, self.spatial_merge_size)?;
                
                let ds_norm_w = self.weights.get(&format!("deepstack.{}.norm.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.norm.weight not found", i))?;
                let ds_norm_b = self.weights.get(&format!("deepstack.{}.norm.bias", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.norm.bias not found", i))?;
                let ds_fc1_w = self.weights.get(&format!("deepstack.{}.fc1.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc1.weight not found", i))?;
                let ds_fc1_b = self.weights.get(&format!("deepstack.{}.fc1.bias", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc1.bias not found", i))?;
                let ds_fc2_w = self.weights.get(&format!("deepstack.{}.fc2.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc2.weight not found", i))?;
                let ds_fc2_b = self.weights.get(&format!("deepstack.{}.fc2.bias", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc2.bias not found", i))?;

                let ds_x = candle_nn::ops::layer_norm(&merged, ds_norm_w, ds_norm_b, 1e-6)?;
                let ds_x = ds_x.matmul(&ds_fc1_w.t()?)?.broadcast_add(ds_fc1_b)?;
                let ds_x = ds_x.gelu()?;
                let ds_x = ds_x.matmul(&ds_fc2_w.t()?)?.broadcast_add(ds_fc2_b)?;

                deepstack_outputs.push(ds_x);
            }
        }

        // 4. Post LN & Projector
        let post_ln_w = self.weights.get("vision.post_ln.weight")
            .ok_or_else(|| anyhow!("vision.post_ln.weight not found"))?;
        let post_ln_b = self.weights.get("vision.post_ln.bias")
            .ok_or_else(|| anyhow!("vision.post_ln.bias not found"))?;
        let x_ln = candle_nn::ops::layer_norm(&x, post_ln_w, post_ln_b, 1e-6)?;

        // Spatial Merge final output
        let merged = spatial_merge(&x_ln, self.spatial_merge_size)?;

        // Main Projector
        let mm_0_w = self.weights.get("projector.0.weight")
            .ok_or_else(|| anyhow!("projector.0.weight not found"))?;
        let mm_0_b = self.weights.get("projector.0.bias")
            .ok_or_else(|| anyhow!("projector.0.bias not found"))?;
        let mm_2_w = self.weights.get("projector.2.weight")
            .ok_or_else(|| anyhow!("projector.2.weight not found"))?;
        let mm_2_b = self.weights.get("projector.2.bias")
            .ok_or_else(|| anyhow!("projector.2.bias not found"))?;

        let proj = merged.matmul(&mm_0_w.t()?)?.broadcast_add(mm_0_b)?;
        let proj = proj.gelu()?;
        let proj = proj.matmul(&mm_2_w.t()?)?.broadcast_add(mm_2_b)?;

        if !deepstack_outputs.is_empty() {
            let mut all_tensors = vec![proj];
            all_tensors.extend(deepstack_outputs);
            let out = Tensor::cat(&all_tensors, 2)?;
            Ok(out)
        } else {
            Ok(proj)
        }
    }
}

fn spatial_merge(x: &Tensor, merge_size: usize) -> Result<Tensor> {
    if merge_size <= 1 {
        return Ok(x.clone());
    }
    let (b, n_patches, dim) = x.dims3()?;
    let grid_size = (n_patches as f64).sqrt() as usize;
    let new_grid_size = grid_size / merge_size;

    let x_reshaped = x.reshape((b, new_grid_size, merge_size, new_grid_size, merge_size, dim))?;
    let x_permuted = x_reshaped.permute((0, 1, 3, 2, 4, 5))?;
    let merged = x_permuted.reshape((b, new_grid_size * new_grid_size, merge_size * merge_size * dim))?;
    Ok(merged)
}

fn normalize_vision_tensors(
    raw: HashMap<String, Tensor>,
    num_layers: usize,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut normalized = HashMap::new();

    // 1. Identify format: check if keys follow GGUF mmproj naming conventions
    let is_gguf = raw.keys().any(|k| k.starts_with("v.") || k.starts_with("mm."));

    if is_gguf {
        for (k, v) in raw {
            if k == "v.patch_embd.weight" {
                normalized.insert("vision.patch_embed.weight".to_string(), v);
            } else if k == "v.patch_embd.bias" {
                normalized.insert("vision.patch_embed.bias".to_string(), v);
            } else if k == "v.position_embd.weight" {
                normalized.insert("vision.pos_embed.weight".to_string(), v);
            } else if k == "v.post_ln.weight" {
                normalized.insert("vision.post_ln.weight".to_string(), v);
            } else if k == "v.post_ln.bias" {
                normalized.insert("vision.post_ln.bias".to_string(), v);
            } else if k == "mm.0.weight" {
                normalized.insert("projector.0.weight".to_string(), v);
            } else if k == "mm.0.bias" {
                normalized.insert("projector.0.bias".to_string(), v);
            } else if k == "mm.2.weight" {
                normalized.insert("projector.2.weight".to_string(), v);
            } else if k == "mm.2.bias" {
                normalized.insert("projector.2.bias".to_string(), v);
            } else if k.starts_with("v.blk.") {
                // e.g. v.blk.0.ln1.weight
                let parts: Vec<&str> = k.split('.').collect();
                if parts.len() >= 5 {
                    let layer: usize = parts[2].parse()?;
                    let component = parts[3];
                    let param = parts[4];
                    let norm_key = format!("vision.layers.{}.{}.{}", layer, component, param);
                    normalized.insert(norm_key, v);
                }
            } else if k.starts_with("v.deepstack.") {
                // e.g. v.deepstack.5.norm.weight
                let parts: Vec<&str> = k.split('.').collect();
                if parts.len() >= 5 {
                    let layer: usize = parts[2].parse()?;
                    let component = parts[3];
                    let param = parts[4];
                    let norm_key = format!("deepstack.{}.{}.{}", layer, component, param);
                    normalized.insert(norm_key, v);
                }
            }
        }
    } else {
        // HF Format
        // Group q/k/v weights to merge them into a single attn_qkv tensor
        let mut q_weights = HashMap::new();
        let mut k_weights = HashMap::new();
        let mut v_weights = HashMap::new();
        let mut q_biases = HashMap::new();
        let mut k_biases = HashMap::new();
        let mut v_biases = HashMap::new();

        for (k, v) in raw {
            if k.contains("patch_embedding.weight") {
                normalized.insert("vision.patch_embed.weight".to_string(), v);
            } else if k.contains("patch_embedding.bias") {
                normalized.insert("vision.patch_embed.bias".to_string(), v);
            } else if k.contains("position_embedding.weight") {
                normalized.insert("vision.pos_embed.weight".to_string(), v);
            } else if k.contains("post_layernorm.weight") {
                normalized.insert("vision.post_ln.weight".to_string(), v);
            } else if k.contains("post_layernorm.bias") {
                normalized.insert("vision.post_ln.bias".to_string(), v);
            } else if k.contains("multi_modal_projector.linear_1.weight") || k.contains("multi_modal_projector.0.weight") {
                normalized.insert("projector.0.weight".to_string(), v);
            } else if k.contains("multi_modal_projector.linear_1.bias") || k.contains("multi_modal_projector.0.bias") {
                normalized.insert("projector.0.bias".to_string(), v);
            } else if k.contains("multi_modal_projector.linear_2.weight") || k.contains("multi_modal_projector.2.weight") {
                normalized.insert("projector.2.weight".to_string(), v);
            } else if k.contains("multi_modal_projector.linear_2.bias") || k.contains("multi_modal_projector.2.bias") {
                normalized.insert("projector.2.bias".to_string(), v);
            } else if k.contains("encoder.layers.") {
                // Find layer index
                let parts: Vec<&str> = k.split('.').collect();
                if let Some(idx_str) = parts.iter().position(|&p| p == "layers").and_then(|pos| parts.get(pos + 1)) {
                    if let Ok(layer_idx) = idx_str.parse::<usize>() {
                        if k.contains("layer_norm1.weight") {
                            normalized.insert(format!("vision.layers.{}.ln1.weight", layer_idx), v);
                        } else if k.contains("layer_norm1.bias") {
                            normalized.insert(format!("vision.layers.{}.ln1.bias", layer_idx), v);
                        } else if k.contains("layer_norm2.weight") {
                            normalized.insert(format!("vision.layers.{}.ln2.weight", layer_idx), v);
                        } else if k.contains("layer_norm2.bias") {
                            normalized.insert(format!("vision.layers.{}.ln2.bias", layer_idx), v);
                        } else if k.contains("self_attn.out_proj.weight") || k.contains("self_attn.dense.weight") {
                            normalized.insert(format!("vision.layers.{}.attn_out.weight", layer_idx), v);
                        } else if k.contains("self_attn.out_proj.bias") || k.contains("self_attn.dense.bias") {
                            normalized.insert(format!("vision.layers.{}.attn_out.bias", layer_idx), v);
                        } else if k.contains("mlp.fc1.weight") || k.contains("mlp.dense_h_to_4h.weight") {
                            normalized.insert(format!("vision.layers.{}.ffn_up.weight", layer_idx), v);
                        } else if k.contains("mlp.fc1.bias") || k.contains("mlp.dense_h_to_4h.bias") {
                            normalized.insert(format!("vision.layers.{}.ffn_up.bias", layer_idx), v);
                        } else if k.contains("mlp.fc2.weight") || k.contains("mlp.dense_4h_to_h.weight") {
                            normalized.insert(format!("vision.layers.{}.ffn_down.weight", layer_idx), v);
                        } else if k.contains("mlp.fc2.bias") || k.contains("mlp.dense_4h_to_h.bias") {
                            normalized.insert(format!("vision.layers.{}.ffn_down.bias", layer_idx), v);
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

        // Concatenate separate Q, K, V projections into a unified attn_qkv tensor
        for layer_idx in 0..num_layers {
            if let (Some(q), Some(k), Some(v)) = (q_weights.remove(&layer_idx), k_weights.remove(&layer_idx), v_weights.remove(&layer_idx)) {
                let qkv = Tensor::cat(&[&q, &k, &v], 0)?;
                normalized.insert(format!("vision.layers.{}.attn_qkv.weight", layer_idx), qkv);
            }
            let q_b = q_biases.remove(&layer_idx);
            let k_b = k_biases.remove(&layer_idx);
            let v_b = v_biases.remove(&layer_idx);
            if q_b.is_some() || k_b.is_some() || v_b.is_some() {
                let q_b = q_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
                let k_b = k_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
                let v_b = v_b.unwrap_or(Tensor::zeros(1, DType::F16, device)?);
                let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;
                normalized.insert(format!("vision.layers.{}.attn_qkv.bias", layer_idx), qkv_b);
            } else {
                // If there are no biases, insert a zero bias tensor of correct length
                if let Some(qkv_w) = normalized.get(&format!("vision.layers.{}.attn_qkv.weight", layer_idx)) {
                    let out_dim = qkv_w.dim(0)?;
                    let zeros = Tensor::zeros(out_dim, DType::F16, device)?;
                    normalized.insert(format!("vision.layers.{}.attn_qkv.bias", layer_idx), zeros);
                }
            }
        }
    }

    Ok(normalized)
}

pub fn load_image(path: &Path, image_size: usize, device: &Device) -> Result<Tensor> {
    tracing::info!("Loading image from {:?} resized to {}", path, image_size);

    let img = image::io::Reader::open(path)?
        .decode()
        .context("Failed to decode image file")?
        .resize_exact(image_size as u32, image_size as u32, image::imageops::FilterType::Triangle);

    let rgb = img.to_rgb8();
    let mut data = vec![0.0f32; 3 * image_size * image_size];

    // Standard Normalization: mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]
    for y in 0..image_size {
        for x in 0..image_size {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            data[0 * image_size * image_size + y * image_size + x] = ((pixel[0] as f32 / 255.0) - 0.5) / 0.5;
            data[1 * image_size * image_size + y * image_size + x] = ((pixel[1] as f32 / 255.0) - 0.5) / 0.5;
            data[2 * image_size * image_size + y * image_size + x] = ((pixel[2] as f32 / 255.0) - 0.5) / 0.5;
        }
    }

    let t = Tensor::from_vec(data, (1, 3, image_size, image_size), device)?;
    Ok(t)
}
