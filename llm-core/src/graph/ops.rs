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
    DeepStackFuse {
        input: String,
        layer_idx: usize,
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
