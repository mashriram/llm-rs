use std::path::Path;
use anyhow::Result;
use tokenizers::Tokenizer;

pub struct LlmTokenizer {
    tokenizer: Tokenizer,
}

impl LlmTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Load a tokenizer from raw bytes (JSON data).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer from bytes: {}", e))?;
        Ok(Self { tokenizer })
    }

    /// Load a tokenizer from a JSON string.
    pub fn from_str(json: &str) -> Result<Self> {
        Self::from_bytes(json.as_bytes())
    }

    /// Encode input text into token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to encode text: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back into a string.
    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer.decode(tokens, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))
    }

    /// Get the vocabulary size of the tokenizer.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get the token ID for a specific token string if it exists in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }
}

