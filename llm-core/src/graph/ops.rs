use crate::types::HiddenAct;

#[derive(Debug, Clone)]
pub enum Operator {
    Embed {
        input_ids: String,
        weight: String,
        output: String,
    },
    RMSNorm {
        input: String,
        weight: String,
        output: String,
        eps: f32,
    },
    MatMul {
        input: String,
        weight: String,
        bias: Option<String>,
        output: String,
    },
    Rope {
        q: String,
        k: String,
        output_q: String,
        output_k: String,
        layer_idx: usize,
        rope_theta: f32,
    },
    RopeQ {
        q: String,
        output_q: String,
        layer_idx: usize,
        rope_theta: f32,
    },
    RopeSkip {
        q: String,
        k: String,
        output_q: String,
        output_k: String,
    },
    PagedAttention {
        q: String,
        k: String,
        v: String,
        output: String,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    },
    Activation {
        input: String,
        output: String,
        act: HiddenAct,
    },
    Mul {
        lhs: String,
        rhs: String,
        output: String,
    },
    Add {
        lhs: String,
        rhs: String,
        output: String,
    },
    VisualEmbed {
        pixel_values: String,
        output: String,
    },
    SpliceTensors {
        text_embeds: String,
        visual_embeds: String,
        output: String,
    },
    AudioEmbed {
        audio_values: String,
        output: String,
    },
    SpliceAudioTensors {
        text_embeds: String,
        audio_embeds: String,
        output: String,
    },
    DeepStackFuse {
        input: String,
        layer_idx: usize,
        output: String,
    },
    Softcap {
        input: String,
        output: String,
        cap: f32,
    },
    Scale {
        input: String,
        scale: f32,
        output: String,
    },
    TensorScale {
        input: String,
        scale_tensor: String,
        output: String,
    },
    PleInput {
        input_ids: String,
        text_embeddings: String,
        per_layer_token_embd: String,
        per_layer_model_proj: String,
        per_layer_proj_norm: String,
        output: String,
    },
    PleLayer {
        input: String,
        per_layer_input: String,
        layer_idx: usize,
        per_layer_input_gate: String,
        per_layer_projection: String,
        post_per_layer_input_norm: String,
        output: String,
    },
}

#[derive(Debug, Clone)]
pub struct ComputeGraph {
    pub ops: Vec<Operator>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn add_op(&mut self, op: Operator) {
        self.ops.push(op);
    }
}
