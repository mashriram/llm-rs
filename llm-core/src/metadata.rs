use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::types::ModelMeta;

#[derive(Debug, Deserialize, Serialize)]
pub struct HfConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub intermediate_size: usize,
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_torch_dtype() -> String {
    "float16".to_string()
}

pub fn parse_metadata<P: AsRef<Path>>(config_path: P) -> Result<ModelMeta> {
    crate::model::config::parse_config(config_path.as_ref())
}
