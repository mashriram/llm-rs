use std::path::Path;
use std::fs::File;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use crate::types::WeightDtype;

/// Byte range + parsed metadata for one tensor, relative to the owning
/// `SafeTensorsFile`'s mmap. Storing offsets instead of a borrowed slice
/// avoids needing a self-referential struct: `tensor()` slices `&self._mmap`
/// on demand, so the returned `SafeTensorView` can never outlive the mmap
/// that backs it — the borrow checker enforces it, no `unsafe` required.
struct StoredTensorMeta {
    offset: usize,
    len: usize,
    shape: Vec<usize>,
    dtype: WeightDtype,
}

pub struct SafeTensorsFile {
    _mmap: Mmap,
    tensor_meta: HashMap<String, StoredTensorMeta>,
}

pub struct SafeTensorView<'a> {
    pub data: &'a [u8],
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

impl SafeTensorsFile {
    /// Borrow one tensor's data directly from the mmap. The returned
    /// `SafeTensorView<'_>` cannot outlive `self`, so it's impossible to hold
    /// a dangling reference after the file (and its mmap) is dropped.
    pub fn tensor(&self, name: &str) -> Option<SafeTensorView<'_>> {
        let meta = self.tensor_meta.get(name)?;
        Some(SafeTensorView {
            data: &self._mmap[meta.offset..meta.offset + meta.len],
            shape: meta.shape.clone(),
            dtype: meta.dtype,
        })
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensor_meta.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensor_meta.keys().map(|s| s.as_str())
    }
}

pub fn load_safetensors(path: &Path) -> Result<SafeTensorsFile> {
    let file = File::open(path)?;
    // SAFETY: memory-mapping a file is unsafe because the mapping becomes invalid
    // (and any read is UB / may SIGBUS) if the underlying file is truncated or
    // otherwise modified by another process while it is mapped. We accept this
    // standard mmap caveat here: `path` is expected to be a stable model weights
    // file that is not concurrently written to while the engine is loading it.
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    let base_ptr = mmap.as_ptr();
    let data_slice: &[u8] = &mmap;

    // `SafeTensors::deserialize` borrows from `data_slice` for the duration of
    // this function only — we extract plain (offset, len) pairs below instead
    // of keeping the parsed `SafeTensors<'_>` (or any borrow derived from it)
    // alive past this scope, so no lifetime erasure is needed.
    let safetensors = SafeTensors::deserialize(data_slice)
        .map_err(|e| anyhow!("Failed to parse safetensors: {}", e))?;

    let mut tensor_meta = HashMap::new();
    for (name, tensor) in safetensors.tensors() {
        let dtype = match tensor.dtype() {
            safetensors::Dtype::F32 => WeightDtype::F32,
            safetensors::Dtype::F16 => WeightDtype::F16,
            safetensors::Dtype::BF16 => WeightDtype::BF16,
            safetensors::Dtype::I8 => WeightDtype::Q8_0, // Map to Q8_0 or similar if needed
            d => return Err(anyhow!("Unsupported SafeTensors dtype: {:?}", d)),
        };

        let data = tensor.data();
        // `data` is a sub-slice of `data_slice` (safetensors::TensorView
        // borrows directly from the buffer passed to `deserialize`), so this
        // pointer-arithmetic offset is in-bounds by construction.
        let offset = (data.as_ptr() as usize) - (base_ptr as usize);

        tensor_meta.insert(name.to_string(), StoredTensorMeta {
            offset,
            len: data.len(),
            shape: tensor.shape().to_vec(),
            dtype,
        });
    }

    Ok(SafeTensorsFile {
        _mmap: mmap,
        tensor_meta,
    })
}
