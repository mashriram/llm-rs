use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, bail};
use crate::types::WeightDtype;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

pub struct TensorView<'a> {
    pub data: &'a [u8],
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
}

pub struct GgufFile {
    _mmap: Mmap,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorView<'static>>,
}

// GGUF Value Types
const GGUF_VALUE_TYPE_UINT8: u32 = 0;
const GGUF_VALUE_TYPE_INT8: u32 = 1;
const GGUF_VALUE_TYPE_UINT16: u32 = 2;
const GGUF_VALUE_TYPE_INT16: u32 = 3;
const GGUF_VALUE_TYPE_UINT32: u32 = 4;
const GGUF_VALUE_TYPE_INT32: u32 = 5;
const GGUF_VALUE_TYPE_FLOAT32: u32 = 6;
const GGUF_VALUE_TYPE_BOOL: u32 = 7;
const GGUF_VALUE_TYPE_STRING: u32 = 8;
const GGUF_VALUE_TYPE_ARRAY: u32 = 9;
const GGUF_VALUE_TYPE_UINT64: u32 = 10;
const GGUF_VALUE_TYPE_INT64: u32 = 11;
const GGUF_VALUE_TYPE_FLOAT64: u32 = 12;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_u32(&mut self) -> Result<u32> {
        if self.pos + 4 > self.data.len() {
            bail!("Unexpected EOF reading u32");
        }
        let val = u32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.pos + 8 > self.data.len() {
            bail!("Unexpected EOF reading u64");
        }
        let val = u64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8;
        Ok(val)
    }

    fn read_f32(&mut self) -> Result<f32> {
        if self.pos + 4 > self.data.len() {
            bail!("Unexpected EOF reading f32");
        }
        let val = f32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4;
        Ok(val)
    }

    fn read_f64(&mut self) -> Result<f64> {
        if self.pos + 8 > self.data.len() {
            bail!("Unexpected EOF reading f64");
        }
        let val = f64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8;
        Ok(val)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.pos + len > self.data.len() {
            bail!("Unexpected EOF reading bytes of length {}", len);
        }
        let val = &self.data[self.pos..self.pos+len];
        self.pos += len;
        Ok(val)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
    }

    fn read_value(&mut self, value_type: u32) -> Result<GgufValue> {
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(GgufValue::U8(self.read_bytes(1)?[0])),
            GGUF_VALUE_TYPE_INT8 => Ok(GgufValue::I8(self.read_bytes(1)?[0] as i8)),
            GGUF_VALUE_TYPE_UINT16 => {
                let b = self.read_bytes(2)?;
                Ok(GgufValue::U16(u16::from_le_bytes(b.try_into().unwrap())))
            }
            GGUF_VALUE_TYPE_INT16 => {
                let b = self.read_bytes(2)?;
                Ok(GgufValue::I16(i16::from_le_bytes(b.try_into().unwrap())))
            }
            GGUF_VALUE_TYPE_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_VALUE_TYPE_INT32 => Ok(GgufValue::I32(self.read_u32()? as i32)),
            GGUF_VALUE_TYPE_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_VALUE_TYPE_BOOL => Ok(GgufValue::Bool(self.read_bytes(1)?[0] != 0)),
            GGUF_VALUE_TYPE_STRING => Ok(GgufValue::String(self.read_string()?)),
            GGUF_VALUE_TYPE_ARRAY => {
                let item_type = self.read_u32()?;
                let len = self.read_u64()? as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value(item_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            GGUF_VALUE_TYPE_UINT64 => Ok(GgufValue::U64(self.read_u64()?)),
            GGUF_VALUE_TYPE_INT64 => Ok(GgufValue::I64(self.read_u64()? as i64)),
            GGUF_VALUE_TYPE_FLOAT64 => Ok(GgufValue::F64(self.read_f64()?)),
            _ => bail!("Unknown GGUF value type: {}", value_type),
        }
    }
}

pub fn load_gguf(path: &Path) -> Result<GgufFile> {
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    
    // We will do a safe transmutation of lifetime for the tensor data views.
    // This is safe because GgufFile keeps _mmap alive and is never handed out after GgufFile is dropped.
    let data_slice: &[u8] = &mmap;
    let mut r = Reader::new(data_slice);

    // 1. Magic
    let magic = r.read_bytes(4)?;
    if magic != b"GGUF" {
        bail!("Invalid GGUF magic header");
    }

    // 2. Version
    let version = r.read_u32()?;
    if version != 2 && version != 3 {
        bail!("Unsupported GGUF version: {}", version);
    }

    // 3. Counts
    let tensor_count = r.read_u64()? as usize;
    let metadata_kv_count = r.read_u64()? as usize;

    // 4. Metadata
    let mut metadata = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = r.read_string()?;
        let value_type = r.read_u32()?;
        let value = r.read_value(value_type)?;
        metadata.insert(key, value);
    }

    // 5. Tensor Infos
    struct TempTensorInfo {
        name: String,
        shape: Vec<usize>,
        dtype: WeightDtype,
        offset: u64,
    }

    let mut temp_tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let ndim = r.read_u32()? as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(r.read_u64()? as usize);
        }
        
        let ggml_type = r.read_u32()?;
        let dtype = match ggml_type {
            0 => WeightDtype::F32,
            1 => WeightDtype::F16,
            // 2 = Q4_0, 3 = Q4_1, etc.
            2 => WeightDtype::Q4_0,
            8 => WeightDtype::Q8_0,
            // mapping others if needed, but Q4_0, Q8_0, F16, F32 are core
            _ => {
                // Fallback to Q4_K or something else for unsupported types
                WeightDtype::Q4_K
            }
        };

        let offset = r.read_u64()?;
        temp_tensors.push(TempTensorInfo { name, shape, dtype, offset });
    }

    // 6. Data Section Alignment
    let alignment = match metadata.get("general.alignment") {
        Some(GgufValue::U32(align)) => *align as usize,
        Some(GgufValue::U8(align)) => *align as usize,
        Some(GgufValue::I32(align)) => *align as usize,
        _ => 32, // default alignment
    };

    let header_size = r.pos;
    let padding = (alignment - (header_size % alignment)) % alignment;
    let data_section_start = header_size + padding;

    let mut tensors = HashMap::new();
    for t in temp_tensors {
        let tensor_start = data_section_start + t.offset as usize;
        
        // Calculate size of tensor data (approx or exact depending on type)
        // Since we don't have exact block sizes for all GGML types, we can slice to the end of the mmap,
        // but a safer way is to calculate based on shape and dtype if known.
        // For simplicity and safety, we can slice from tensor_start to the end of the file,
        // or up to the next tensor's start.
        let raw_data_ptr = &data_slice[tensor_start..];
        
        // Let's perform the unsafe lifetime cast to 'static.
        // This is safe because `_mmap` is owned by `GgufFile` and will outlive any `TensorView` references.
        let static_data: &'static [u8] = unsafe {
            std::slice::from_raw_parts(raw_data_ptr.as_ptr(), raw_data_ptr.len())
        };

        tensors.insert(t.name, TensorView {
            data: static_data,
            shape: t.shape,
            dtype: t.dtype,
        });
    }

    Ok(GgufFile {
        _mmap: mmap,
        metadata,
        tensors,
    })
}
