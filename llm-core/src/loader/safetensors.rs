use std::path::Path;
use std::fs::File;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use memmap2::{Mmap, MmapOptions};
use safetensors::SafeTensors;
use crate::types::WeightDtype;

pub struct SafeTensorsFile {
    _mmap: Mmap,
    pub tensors: HashMap<String, SafeTensorView<'static>>,
}

pub struct SafeTensorView<'a> {
    pub data: &'a [u8],
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

pub fn load_safetensors(path: &Path) -> Result<SafeTensorsFile> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };

    // We do a safe transmutation of lifetime for SafeTensors parsing.
    // The SafeTensors object is parsed from the mmap slice.
    let data_slice: &[u8] = &mmap;
    
    // Safety: we extend the lifetime of the parsed SafeTensors data to 'static
    // because GgufFile/SafeTensorsFile owns the `_mmap` and will outlive any
    // references returned from it.
    let static_data: &'static [u8] = unsafe {
        std::slice::from_raw_parts(data_slice.as_ptr(), data_slice.len())
    };

    let safetensors = SafeTensors::deserialize(static_data)
        .map_err(|e| anyhow!("Failed to parse safetensors: {}", e))?;

    let mut tensors = HashMap::new();
    for (name, tensor) in safetensors.tensors() {
        let dtype = match tensor.dtype() {
            safetensors::Dtype::F32 => WeightDtype::F32,
            safetensors::Dtype::F16 => WeightDtype::F16,
            safetensors::Dtype::BF16 => WeightDtype::BF16,
            safetensors::Dtype::I8 => WeightDtype::Q8_0, // Map to Q8_0 or similar if needed
            d => return Err(anyhow!("Unsupported SafeTensors dtype: {:?}", d)),
        };

        tensors.insert(name.to_string(), SafeTensorView {
            data: tensor.data(),
            shape: tensor.shape().to_vec(),
            dtype,
        });
    }

    Ok(SafeTensorsFile {
        _mmap: mmap,
        tensors,
    })
}
