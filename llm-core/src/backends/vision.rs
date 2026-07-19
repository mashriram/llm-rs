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

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let s = x.sqr()?;
    let sum = s.sum_keepdim(candle_core::D::Minus1)?;
    let last_dim = x.dim(candle_core::D::Minus1)?;
    let mean = (sum / (last_dim as f64))?;
    let norm = (mean + eps)?.sqrt()?;
    let x_normed = x.broadcast_div(&norm)?;
    Ok(x_normed.broadcast_mul(weight)?)
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
        // Known spatial_merge_size by family (verified against upstream configs):
        //   Qwen2-VL family: 2
        //   Gemma-4 vision (SigLIP-derived): 1
        //   default when metadata absent: 1 (safe: produces MORE tokens than needed, never fewer)
        let mut spatial_merge_size = 1;
        let mut is_deepstack_layers = vec![false; 24];

        let is_gguf = path.is_file() && path.extension().and_then(|ext| ext.to_str()).map(|ext| ext.to_lowercase() == "gguf").unwrap_or(false);

        if is_gguf {
            let mut file = std::fs::File::open(path)
                .context(format!("Failed to open mmproj GGUF file: {:?}", path))?;
            let model = candle_core::quantized::gguf_file::Content::read(&mut file)
                .context("Failed to read GGUF mmproj content")?;

            let get_metadata_u32 = |key: &str| -> Option<u32> {
                match model.metadata.get(key) {
                    Some(candle_core::quantized::gguf_file::Value::U8(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::I8(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::U16(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::I16(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::U32(v)) => Some(*v),
                    Some(candle_core::quantized::gguf_file::Value::I32(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::U64(v)) => Some(*v as u32),
                    Some(candle_core::quantized::gguf_file::Value::I64(v)) => Some(*v as u32),
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
            // Determine the working dtype: F32 on CPU for numerical accuracy,
            // F16 on GPU for throughput. The vision encoder always computes on the
            // target device, so we dequantize directly there.
            let vision_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            for name in model.tensor_infos.keys() {
                // Skip quantization scale/min/max tensors (input_max, input_min, output_max, output_min)
                // These are calibration tensors from quantized models, not actual weights.
                let last_part = name.rsplit('.').next().unwrap_or("");
                if matches!(last_part, "input_max" | "input_min" | "output_max" | "output_min") {
                    continue;
                }
                let qtensor = model.tensor(&mut file, name, &cpu)
                    .context(format!("Failed to load mmproj tensor {}", name))?;
                let tensor = qtensor.dequantize(&cpu)
                    .context(format!("Failed to dequantize mmproj tensor {}", name))?
                    .to_dtype(vision_dtype)?
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

            let vision_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            for (k, v) in loaded {
                raw_weights.insert(k, v.to_dtype(vision_dtype)?);
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

        // Prefer deriving spatial_merge_size from the projector's own input
        // dimension over trusting a GGUF metadata key. Confirmed via a real
        // Qwen2-VL-2B-Instruct mmproj file that `clip.vision.spatial_merge_size`
        // is simply absent from real-world exports, which silently left this
        // at the hardcoded default of 1 and caused a matmul shape-mismatch
        // crash on the very first real forward pass (projector expected
        // hidden_dim * merge_size^2 input, e.g. 1280*4=5120, but got the
        // un-merged 1280). The projector's weight shape is ground truth for
        // what the model actually needs — this is metadata (the model file
        // itself), just not a metadata *key* — so use it whenever it cleanly
        // resolves, falling back to the config/GGUF-key value otherwise.
        if let Some(proj_w) = weights.get("projector.0.weight") {
            if hidden_dim > 0 {
                for &dim in &[proj_w.dim(0)?, proj_w.dim(proj_w.rank().saturating_sub(1))?] {
                    if dim > hidden_dim && dim % hidden_dim == 0 {
                        let ratio = dim / hidden_dim;
                        let candidate = (ratio as f64).sqrt().round() as usize;
                        if candidate >= 1 && candidate * candidate == ratio {
                            spatial_merge_size = candidate;
                            break;
                        }
                    }
                }
            }
        }

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

        let mut x = pixel_values.conv2d(patch_emb_w, 0, self.patch_size, 1, 1)?;

        if let Some(bias) = patch_emb_b {
            let bias_reshaped = bias.reshape((1, self.hidden_dim, 1, 1))?;
            x = x.broadcast_add(&bias_reshaped)?;
        }

        // Reshape [1, hidden_dim, H_out, W_out] -> [1, hidden_dim, patches] -> [1, patches, hidden_dim]
        let (b, c, h, w) = x.dims4()?;
        let num_patches = h * w;
        // `.contiguous()` is required here, not optional: `permute` only changes
        // strides, and when a model has no `vision.pos_embed.weight` (e.g.
        // Qwen2-VL, which uses rotary position embeddings for its vision
        // encoder instead of an absolute position table), the `broadcast_add`
        // below that would otherwise produce a contiguous tensor never runs -
        // `x` flows straight into `candle_nn::ops::layer_norm` still permuted,
        // which candle rejects with "Non contiguous layernorm is not
        // implemented" (confirmed via a real Qwen2-VL-2B forward pass).
        x = x.reshape((b, c, num_patches))?.permute((0, 2, 1))?.contiguous()?;

        // 2. Add Positional Embedding (pos_embed.weight shape [num_patches, hidden_dim] or [2, 10240, hidden_dim])
        if let Some(pos_emb) = self.weights.get("vision.pos_embed.weight") {
            let pos_shape = pos_emb.shape();
            if pos_shape.dims().len() == 3 && pos_shape.dims()[0] == 2 {
                // Factorized 2D position embedding lookup
                let y_table = pos_emb.get(0)?.narrow(0, 0, h)?; // Shape [h, c]
                let x_table = pos_emb.get(1)?.narrow(0, 0, w)?; // Shape [w, c]
                let y_reshaped = y_table.reshape((h, 1, c))?;
                let x_reshaped = x_table.reshape((1, w, c))?;
                let grid_pos = y_reshaped.broadcast_add(&x_reshaped)?; // Shape [h, w, c]
                let pos_to_add = grid_pos.reshape((1, num_patches, c))?;
                x = x.broadcast_add(&pos_to_add)?;
            } else {
                // 1D learned absolute position embedding
                let pos_len = pos_emb.dim(0)?;
                // Take the FIRST `num_patches` rows of the position embedding table
                // (position 0 must align with patch 0). Slicing from the end
                // (`pos_len - num_patches`) was an off-by-one/wrong-end bug: it
                // would apply the tail of a longer table's positions to the
                // beginning of the patch sequence, misaligning every position.
                let pos_to_add = if pos_len > num_patches {
                    pos_emb.narrow(0, 0, num_patches)?
                } else {
                    pos_emb.clone()
                };
                x = x.broadcast_add(&pos_to_add.reshape((1, num_patches, c))?)?;
            }
        }

        // Fuyu/Unified Bypass: if num_layers is 0, we bypass the trunk entirely and project directly!
        if self.num_layers == 0 {
            // Direct projection
            if let Some(mm_0_w) = self.weights.get("projector.0.weight") {
                let mm_0_b = self.weights.get("projector.0.bias");
                x = matmul_3d_2d(&x, &mm_0_w.t()?)?;
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
        // Determine working dtype from the patch embedding weight
        let work_dtype = patch_emb_w.dtype();

        for i in 0..self.num_layers {
            // LayerNorm 1
            let ln1_w = self.weights.get(&format!("vision.layers.{}.ln1.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln1.weight not found", i))?;
            let ln1_b = self.weights.get(&format!("vision.layers.{}.ln1.bias", i));
            let x_ln1 = if let Some(bias) = ln1_b {
                candle_nn::ops::layer_norm(&x, ln1_w, bias, 1e-6)?
            } else {
                rms_norm(&x, ln1_w, 1e-6)?
            };

            // Self Attention QKV
            let qkv_w = self.weights.get(&format!("vision.layers.{}.attn_qkv.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_qkv.weight not found", i))?;
            let qkv_b = self.weights.get(&format!("vision.layers.{}.attn_qkv.bias", i));

            let mut qkv = matmul_3d_2d(&x_ln1, qkv_w)?;
            if let Some(bias) = qkv_b {
                qkv = qkv.broadcast_add(bias)?;
            }
            let qkv_chunks = qkv.chunk(3, 2)?;
            let mut q = qkv_chunks[0].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;
            let mut k = qkv_chunks[1].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;
            let v = qkv_chunks[2].reshape((1, num_patches, self.num_heads, head_dim))?.permute((0, 2, 1, 3))?;

            // QK-Norm
            let q_norm_w = self.weights.get(&format!("vision.layers.{}.attn_q_norm.weight", i));
            let k_norm_w = self.weights.get(&format!("vision.layers.{}.attn_k_norm.weight", i));
            if let (Some(qw), Some(kw)) = (q_norm_w, k_norm_w) {
                q = rms_norm(&q, qw, 1e-6)?;
                k = rms_norm(&k, kw, 1e-6)?;
            }

            let attn_weights = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
            let attn_probs = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;
            let attn_out = attn_probs.matmul(&v.contiguous()?)?;

            let attn_out = attn_out.permute((0, 2, 1, 3))?.contiguous()?.reshape((1, num_patches, self.hidden_dim))?;

            // Output projection
            let out_w = self.weights.get(&format!("vision.layers.{}.attn_out.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.attn_out.weight not found", i))?;
            let out_b = self.weights.get(&format!("vision.layers.{}.attn_out.bias", i));
            let mut attn_out_proj = matmul_3d_2d(&attn_out, out_w)?.to_dtype(work_dtype)?;
            if let Some(bias) = out_b {
                attn_out_proj = attn_out_proj.broadcast_add(&bias.to_dtype(work_dtype)?)?;
            }

            // Apply post-attention norm if present (Gemma-4 style)
            if let Some(post_norm_w) = self.weights.get(&format!("vision.layers.{}.attn_post_norm.weight", i)) {
                attn_out_proj = rms_norm(&attn_out_proj, &post_norm_w.to_dtype(work_dtype)?, 1e-6)?;
            }

            x = (x.to_dtype(work_dtype)? + attn_out_proj)?;

            // LayerNorm 2
            let ln2_w = self.weights.get(&format!("vision.layers.{}.ln2.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ln2.weight not found", i))?;
            let ln2_b = self.weights.get(&format!("vision.layers.{}.ln2.bias", i));
            let x_ln2 = if let Some(bias) = ln2_b {
                candle_nn::ops::layer_norm(&x, ln2_w, bias, 1e-6)?
            } else {
                rms_norm(&x, ln2_w, 1e-6)?
            };

            // MLP (Gated GeGLU or Standard)
            let ffn_up_w = self.weights.get(&format!("vision.layers.{}.ffn_up.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_up.weight not found", i))?;
            let ffn_up_b = self.weights.get(&format!("vision.layers.{}.ffn_up.bias", i));
            let ffn_down_w = self.weights.get(&format!("vision.layers.{}.ffn_down.weight", i))
                .ok_or_else(|| anyhow!("vision.layers.{}.ffn_down.weight not found", i))?;
            let ffn_down_b = self.weights.get(&format!("vision.layers.{}.ffn_down.bias", i));

            let ffn_gate_w = self.weights.get(&format!("vision.layers.{}.ffn_gate.weight", i));
            let ffn_gate_b = self.weights.get(&format!("vision.layers.{}.ffn_gate.bias", i));

            let mlp = if let Some(gate_w) = ffn_gate_w {
                // Gated MLP (GeGLU)
                let up = add_bias_if_matching(matmul_3d_2d(&x_ln2, ffn_up_w)?, ffn_up_b, "ffn_up")?;
                let gate = add_bias_if_matching(matmul_3d_2d(&x_ln2, gate_w)?, ffn_gate_b, "ffn_gate")?;
                let gate_act = gate.gelu()?;
                (gate_act * up)?
            } else {
                // Standard MLP
                let up = add_bias_if_matching(matmul_3d_2d(&x_ln2, ffn_up_w)?, ffn_up_b, "ffn_up")?;
                up.gelu()?
            };

            let mut mlp_out = add_bias_if_matching(
                matmul_3d_2d(&mlp, ffn_down_w)?.to_dtype(work_dtype)?,
                ffn_down_b.map(|b| b.to_dtype(work_dtype)).transpose()?.as_ref(),
                "ffn_down",
            )?;

            // Apply post-FFN norm if present (Gemma-4 style)
            if let Some(post_norm_w) = self.weights.get(&format!("vision.layers.{}.ffn_post_norm.weight", i)) {
                mlp_out = rms_norm(&mlp_out, &post_norm_w.to_dtype(work_dtype)?, 1e-6)?;
            }

            x = (x.to_dtype(work_dtype)? + mlp_out)?;

            // DeepStack integration
            if self.is_deepstack_layers.get(i).copied().unwrap_or(false) {
                let merged = spatial_merge(&x, self.spatial_merge_size)?;
                
                let ds_norm_w = self.weights.get(&format!("deepstack.{}.norm.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.norm.weight not found", i))?;
                let ds_norm_b = self.weights.get(&format!("deepstack.{}.norm.bias", i));
                let ds_fc1_w = self.weights.get(&format!("deepstack.{}.fc1.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc1.weight not found", i))?;
                let ds_fc1_b = self.weights.get(&format!("deepstack.{}.fc1.bias", i));
                let ds_fc2_w = self.weights.get(&format!("deepstack.{}.fc2.weight", i))
                    .ok_or_else(|| anyhow!("deepstack.{}.fc2.weight not found", i))?;
                let ds_fc2_b = self.weights.get(&format!("deepstack.{}.fc2.bias", i));

                let ds_norm = if let Some(bias) = ds_norm_b {
                    candle_nn::ops::layer_norm(&merged, ds_norm_w, bias, 1e-6)?
                } else {
                    rms_norm(&merged, ds_norm_w, 1e-6)?
                };

                let mut ds_fc1 = matmul_3d_2d(&ds_norm, ds_fc1_w)?;
                if let Some(bias) = ds_fc1_b {
                    ds_fc1 = ds_fc1.broadcast_add(bias)?;
                }
                let ds_fc1 = ds_fc1.gelu()?;

                let mut ds_fc2 = matmul_3d_2d(&ds_fc1, ds_fc2_w)?;
                if let Some(bias) = ds_fc2_b {
                    ds_fc2 = ds_fc2.broadcast_add(bias)?;
                }

                deepstack_outputs.push(ds_fc2);
            }
        }

        // 4. Post LN & Projector
        let x_ln = if let Some(post_ln_w) = self.weights.get("vision.post_ln.weight") {
            let post_ln_b = self.weights.get("vision.post_ln.bias");
            if let Some(bias) = post_ln_b {
                candle_nn::ops::layer_norm(&x, post_ln_w, bias, 1e-6)?
            } else {
                rms_norm(&x, post_ln_w, 1e-6)?
            }
        } else {
            x.clone()
        };

        // Spatial Merge final output
        let merged = spatial_merge(&x_ln, self.spatial_merge_size)?;

        // Main Projector (supports both 1-layer and 2-layer MLP projectors)
        let mm_0_w = self.weights.get("projector.0.weight")
            .ok_or_else(|| anyhow!("projector.0.weight not found"))?;
        let mm_0_b = self.weights.get("projector.0.bias");
 
        let mut proj = matmul_3d_2d(&merged, &mm_0_w.t()?)?;
        if let Some(b) = mm_0_b {
            proj = proj.broadcast_add(b)?;
        }
 
        if let Some(mm_2_w) = self.weights.get("projector.2.weight") {
            let mm_2_b = self.weights.get("projector.2.bias");
            proj = proj.gelu()?;
            proj = matmul_3d_2d(&proj, &mm_2_w.t()?)?;
            if let Some(b) = mm_2_b {
                proj = proj.broadcast_add(b)?;
            }
        }

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
            } else if k == "mm.input_projection.weight" || k == "mm.0.weight" {
                normalized.insert("projector.0.weight".to_string(), v);
            } else if k == "mm.input_projection.bias" || k == "mm.0.bias" {
                normalized.insert("projector.0.bias".to_string(), v);
            } else if k == "mm.2.weight" {
                normalized.insert("projector.2.weight".to_string(), v);
            } else if k == "mm.2.bias" {
                normalized.insert("projector.2.bias".to_string(), v);
            } else if k.starts_with("v.blk.") {
                // e.g. v.blk.0.ln1.weight  OR  v.blk.0.attn_post_norm.weight
                let parts: Vec<&str> = k.split('.').collect();
                // Skip quantization calibration tensors (e.g. v.blk.0.attn_k.input_max)
                let last = parts.last().copied().unwrap_or("");
                if matches!(last, "input_max" | "input_min" | "output_max" | "output_min") {
                    continue;
                }
                if parts.len() >= 5 {
                    let layer_opt = parts[2].parse::<usize>();
                    if let Ok(layer) = layer_opt {
                        let component = parts[3];
                        let param = parts[4];
                        // Skip relative position bias (not used in our forward pass)
                        if component.ends_with("_rel") {
                            continue;
                        }
                        let norm_key = format!("vision.layers.{}.{}.{}", layer, component, param);
                        normalized.insert(norm_key, v);
                    }
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

        // Insert grouped projections back for processing
        for (layer_idx, v) in q_weights {
            normalized.insert(format!("vision.layers.{}.attn_q.weight", layer_idx), v);
        }
        for (layer_idx, v) in k_weights {
            normalized.insert(format!("vision.layers.{}.attn_k.weight", layer_idx), v);
        }
        for (layer_idx, v) in v_weights {
            normalized.insert(format!("vision.layers.{}.attn_v.weight", layer_idx), v);
        }
        for (layer_idx, v) in q_biases {
            normalized.insert(format!("vision.layers.{}.attn_q.bias", layer_idx), v);
        }
        for (layer_idx, v) in k_biases {
            normalized.insert(format!("vision.layers.{}.attn_k.bias", layer_idx), v);
        }
        for (layer_idx, v) in v_biases {
            normalized.insert(format!("vision.layers.{}.attn_v.bias", layer_idx), v);
        }
    }

    // Shared post-processing to group and concatenate separate Q, K, V projections
    let mut q_weights = HashMap::new();
    let mut k_weights = HashMap::new();
    let mut v_weights = HashMap::new();
    let mut q_biases = HashMap::new();
    let mut k_biases = HashMap::new();
    let mut v_biases = HashMap::new();

    let keys: Vec<String> = normalized.keys().cloned().collect();
    for k in keys {
        if k.starts_with("vision.layers.") {
            let parts: Vec<&str> = k.split('.').collect();
            if parts.len() >= 5 {
                if let Ok(layer_idx) = parts[2].parse::<usize>() {
                    let component = parts[3];
                    let param = parts[4];
                    if component == "attn_q" && param == "weight" {
                        if let Some(v) = normalized.remove(&k) {
                            q_weights.insert(layer_idx, v);
                        }
                    } else if component == "attn_k" && param == "weight" {
                        if let Some(v) = normalized.remove(&k) {
                            k_weights.insert(layer_idx, v);
                        }
                    } else if component == "attn_v" && param == "weight" {
                        if let Some(v) = normalized.remove(&k) {
                            v_weights.insert(layer_idx, v);
                        }
                    } else if component == "attn_q" && param == "bias" {
                        if let Some(v) = normalized.remove(&k) {
                            q_biases.insert(layer_idx, v);
                        }
                    } else if component == "attn_k" && param == "bias" {
                        if let Some(v) = normalized.remove(&k) {
                            k_biases.insert(layer_idx, v);
                        }
                    } else if component == "attn_v" && param == "bias" {
                        if let Some(v) = normalized.remove(&k) {
                            v_biases.insert(layer_idx, v);
                        }
                    }
                }
            }
        }
    }

    let vision_dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
    for layer_idx in 0..num_layers {
        let q_w = q_weights.remove(&layer_idx);
        let k_w = k_weights.remove(&layer_idx);
        let v_w = v_weights.remove(&layer_idx);
        // Remember each projection's actual output dimension (from its weight)
        // before the weights are consumed below, so a missing bias for one of
        // Q/K/V can be zero-filled at the CORRECT shape instead of a bogus
        // length-1 placeholder that would break `broadcast_add` downstream.
        //
        // These GGUF vision weights load with dims() = [in_features,
        // out_features] (confirmed empirically against a real Qwen2-VL-2B
        // mmproj file: `ffn_up.weight` loads as [1280, 5120] = [in, out], not
        // PyTorch's usual [out, in] — candle's GGUF loader for this tensor
        // family does not reverse GGUF's native axis order the way it does
        // for e.g. conv weights). So the output axis is dim 1, and matmul
        // call sites use these weights directly with no `.t()` — see
        // `matmul_3d_2d`, which expects rhs already as [in, out].
        let q_out_dim = q_w.as_ref().map(|t| t.dim(1)).transpose()?;
        let k_out_dim = k_w.as_ref().map(|t| t.dim(1)).transpose()?;
        let v_out_dim = v_w.as_ref().map(|t| t.dim(1)).transpose()?;
        if let (Some(q), Some(k), Some(v)) = (&q_w, &k_w, &v_w) {
            // Concatenate along the OUTPUT axis (dim 1, not dim 0 — dim 0 is
            // the shared input/hidden dimension, which must stay intact for
            // the fused [in, 3*out] projection to be meaningful).
            let qkv = Tensor::cat(&[q, k, v], 1)?;
            normalized.insert(format!("vision.layers.{}.attn_qkv.weight", layer_idx), qkv);
        }
        let q_b = q_biases.remove(&layer_idx);
        let k_b = k_biases.remove(&layer_idx);
        let v_b = v_biases.remove(&layer_idx);
        if q_b.is_some() || k_b.is_some() || v_b.is_some() {
            let q_b = match q_b {
                Some(t) => t,
                None => Tensor::zeros(q_out_dim.unwrap_or(1), vision_dtype, device)?,
            };
            let k_b = match k_b {
                Some(t) => t,
                None => Tensor::zeros(k_out_dim.unwrap_or(1), vision_dtype, device)?,
            };
            let v_b = match v_b {
                Some(t) => t,
                None => Tensor::zeros(v_out_dim.unwrap_or(1), vision_dtype, device)?,
            };
            let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;
            normalized.insert(format!("vision.layers.{}.attn_qkv.bias", layer_idx), qkv_b);
        } else {
            // Generate zero bias if the weight was successfully created/found
            if let Some(qkv_w) = normalized.get(&format!("vision.layers.{}.attn_qkv.weight", layer_idx)) {
                let out_dim = qkv_w.dim(1)?;
                let zeros = Tensor::zeros(out_dim, vision_dtype, device)?;
                normalized.insert(format!("vision.layers.{}.attn_qkv.bias", layer_idx), zeros);
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
            let plane = image_size * image_size;
            data[y * image_size + x] = ((pixel[0] as f32 / 255.0) - 0.5) / 0.5;
            data[plane + y * image_size + x] = ((pixel[1] as f32 / 255.0) - 0.5) / 0.5;
            data[2 * plane + y * image_size + x] = ((pixel[2] as f32 / 255.0) - 0.5) / 0.5;
        }
    }

    let t = Tensor::from_vec(data, (1, 3, image_size, image_size), device)?;
    Ok(t)
}

fn matmul_3d_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    Ok(lhs.contiguous()?.matmul(&rhs.unsqueeze(0)?.contiguous()?)?)
}

/// Add `bias` to the last dimension of `x`, but only if its length actually
/// matches — some real-world GGUF mmproj exports carry a bias tensor whose
/// shape doesn't match its own weight's output dimension (confirmed against
/// a real Qwen2-VL-2B-Instruct mmproj file: `ffn_up.bias` is `[1280]` while
/// `ffn_up.weight`'s output dimension is `5120`, an internally-inconsistent
/// export). Rather than crash on `broadcast_add`'s shape mismatch, skip a
/// bias that doesn't fit — matching CLAUDE.md's "no panic on malformed
/// external input" rule — and log it once so the gap is visible, not silent.
fn add_bias_if_matching(x: Tensor, bias: Option<&Tensor>, what: &str) -> Result<Tensor> {
    let Some(bias) = bias else { return Ok(x) };
    let expected = x.dim(x.rank() - 1)?;
    let actual = bias.dim(bias.rank() - 1)?;
    if actual != expected {
        tracing::warn!(
            "{what}: bias shape {:?} doesn't match expected output dim {} \
             (got {}) — this GGUF file's bias tensor appears inconsistent \
             with its own weight; skipping this bias rather than guessing",
            bias.dims(), expected, actual
        );
        return Ok(x);
    }
    Ok(x.broadcast_add(bias)?)
}

