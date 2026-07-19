use memmap2::{Mmap, MmapOptions};
use std::fs::File;
use std::path::Path;
use std::collections::HashMap;
use anyhow::{Result, anyhow, bail, Context};
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

/// Upper bound on any file-derived count (metadata KV pairs, array elements,
/// tensor count, tensor dimensionality). GGUF headers store these as untrusted
/// u32/u64 values read straight from the file; without a sanity cap, a
/// corrupt or malicious file can make us `Vec::with_capacity(huge_number)`
/// and OOM/DoS before any real validation happens. 10 million is generous for
/// any legitimate model file (a 10M-dim tensor axis or 10M metadata KVs would
/// already be nonsensical) while still being cheap to check.
const MAX_REASONABLE_COUNT: u64 = 10_000_000;

/// Cap on nested `GgufValue::Array` recursion depth. GGUF arrays can (per
/// spec) contain arrays of arrays; without a depth cap, a crafted file could
/// force unbounded recursion (stack overflow) via deeply nested empty arrays.
const MAX_ARRAY_NESTING_DEPTH: u32 = 32;

/// Byte offset (to end-of-file, matching this loader's original slicing
/// behavior) + parsed metadata for one tensor, relative to `GgufFile`'s mmap.
/// Storing an offset instead of a borrowed slice avoids a self-referential
/// struct: `tensor()` slices `&self._mmap` on demand, so the returned
/// `TensorView` can never outlive the mmap that backs it — enforced by the
/// borrow checker, no `unsafe` required.
struct StoredTensorMeta {
    offset: usize,
    len: usize,
    shape: Vec<usize>,
    dtype: WeightDtype,
}

/// Number of raw bytes needed to store `elem_count` elements of `dtype`, in
/// GGML's on-disk layout. Used to validate that a declared tensor doesn't
/// claim more data than actually exists between its offset and EOF.
fn dtype_byte_len(dtype: WeightDtype, elem_count: usize) -> Result<usize> {
    const Q_BLOCK_ELEMS: usize = 32;
    match dtype {
        WeightDtype::F32 => Ok(elem_count * 4),
        WeightDtype::F16 | WeightDtype::BF16 => Ok(elem_count * 2),
        WeightDtype::I8 => Ok(elem_count),
        WeightDtype::Q8_0 => {
            // 34 bytes per 32-element block: 2-byte f16 scale + 32 i8 values.
            let blocks = elem_count.div_ceil(Q_BLOCK_ELEMS);
            Ok(blocks * 34)
        }
        WeightDtype::Q4_0 => {
            // 18 bytes per 32-element block: 2-byte f16 scale + 16 bytes of packed 4-bit values.
            let blocks = elem_count.div_ceil(Q_BLOCK_ELEMS);
            Ok(blocks * 18)
        }
        WeightDtype::Q4_K => bail!("Q4_K byte-length validation not implemented in this loader"),
    }
}

pub struct GgufFile {
    _mmap: Mmap,
    pub metadata: HashMap<String, GgufValue>,
    tensor_meta: HashMap<String, StoredTensorMeta>,
}

impl GgufFile {
    /// Borrow one tensor's data directly from the mmap. The returned
    /// `TensorView<'_>` cannot outlive `self`, so it's impossible to hold a
    /// dangling reference after the file (and its mmap) is dropped.
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let meta = self.tensor_meta.get(name)?;
        Some(TensorView {
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
        let val = u32::from_le_bytes(
            self.data[self.pos..self.pos+4]
                .try_into()
                .map_err(|_| anyhow!("GGUF truncated at offset {}: expected 4 bytes for u32", self.pos))?
        );
        self.pos += 4;
        Ok(val)
    }

    fn read_u64(&mut self) -> Result<u64> {
        if self.pos + 8 > self.data.len() {
            bail!("Unexpected EOF reading u64");
        }
        let val = u64::from_le_bytes(
            self.data[self.pos..self.pos+8]
                .try_into()
                .map_err(|_| anyhow!("GGUF truncated at offset {}: expected 8 bytes for u64", self.pos))?
        );
        self.pos += 8;
        Ok(val)
    }

    fn read_f32(&mut self) -> Result<f32> {
        if self.pos + 4 > self.data.len() {
            bail!("Unexpected EOF reading f32");
        }
        let val = f32::from_le_bytes(
            self.data[self.pos..self.pos+4]
                .try_into()
                .map_err(|_| anyhow!("GGUF truncated at offset {}: expected 4 bytes for f32", self.pos))?
        );
        self.pos += 4;
        Ok(val)
    }

    fn read_f64(&mut self) -> Result<f64> {
        if self.pos + 8 > self.data.len() {
            bail!("Unexpected EOF reading f64");
        }
        let val = f64::from_le_bytes(
            self.data[self.pos..self.pos+8]
                .try_into()
                .map_err(|_| anyhow!("GGUF truncated at offset {}: expected 8 bytes for f64", self.pos))?
        );
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
        self.read_value_depth(value_type, 0)
    }

    fn read_value_depth(&mut self, value_type: u32, depth: u32) -> Result<GgufValue> {
        if depth > MAX_ARRAY_NESTING_DEPTH {
            bail!(
                "GGUF array nesting depth {} exceeds max {} — corrupt/malicious file",
                depth, MAX_ARRAY_NESTING_DEPTH
            );
        }
        match value_type {
            GGUF_VALUE_TYPE_UINT8 => Ok(GgufValue::U8(self.read_bytes(1)?[0])),
            GGUF_VALUE_TYPE_INT8 => Ok(GgufValue::I8(self.read_bytes(1)?[0] as i8)),
            GGUF_VALUE_TYPE_UINT16 => {
                let b = self.read_bytes(2)?;
                Ok(GgufValue::U16(u16::from_le_bytes(
                    b.try_into().map_err(|_| anyhow!("GGUF: invalid u16 bytes"))?
                )))
            }
            GGUF_VALUE_TYPE_INT16 => {
                let b = self.read_bytes(2)?;
                Ok(GgufValue::I16(i16::from_le_bytes(
                    b.try_into().map_err(|_| anyhow!("GGUF: invalid i16 bytes"))?
                )))
            }
            GGUF_VALUE_TYPE_UINT32 => Ok(GgufValue::U32(self.read_u32()?)),
            GGUF_VALUE_TYPE_INT32 => Ok(GgufValue::I32(self.read_u32()? as i32)),
            GGUF_VALUE_TYPE_FLOAT32 => Ok(GgufValue::F32(self.read_f32()?)),
            GGUF_VALUE_TYPE_BOOL => Ok(GgufValue::Bool(self.read_bytes(1)?[0] != 0)),
            GGUF_VALUE_TYPE_STRING => Ok(GgufValue::String(self.read_string()?)),
            GGUF_VALUE_TYPE_ARRAY => {
                let item_type = self.read_u32()?;
                let len = self.read_u64()?;
                if len > MAX_REASONABLE_COUNT {
                    bail!(
                        "GGUF array length {} exceeds sanity bound {} — corrupt/malicious file",
                        len, MAX_REASONABLE_COUNT
                    );
                }
                let len = len as usize;
                let mut items = Vec::with_capacity(len);
                for _ in 0..len {
                    items.push(self.read_value_depth(item_type, depth + 1)?);
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
    // SAFETY: memory-mapping a file is unsafe because the mapping becomes invalid
    // (and any read is UB / may SIGBUS) if the underlying file is truncated or
    // otherwise modified by another process while it is mapped. We accept this
    // standard mmap caveat here: `path` is expected to be a stable model weights
    // file that is not concurrently written to while the engine is loading it.
    let mmap = unsafe { MmapOptions::new().map(&file)? };

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
    let tensor_count_raw = r.read_u64()?;
    let metadata_kv_count_raw = r.read_u64()?;
    if tensor_count_raw > MAX_REASONABLE_COUNT {
        bail!(
            "GGUF tensor_count {} exceeds sanity bound {} — corrupt/malicious file",
            tensor_count_raw, MAX_REASONABLE_COUNT
        );
    }
    if metadata_kv_count_raw > MAX_REASONABLE_COUNT {
        bail!(
            "GGUF metadata_kv_count {} exceeds sanity bound {} — corrupt/malicious file",
            metadata_kv_count_raw, MAX_REASONABLE_COUNT
        );
    }
    let tensor_count = tensor_count_raw as usize;
    let metadata_kv_count = metadata_kv_count_raw as usize;

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
        let ndim_raw = r.read_u32()?;
        if ndim_raw as u64 > MAX_REASONABLE_COUNT {
            bail!(
                "GGUF tensor {:?} has ndim {} exceeding sanity bound {} — corrupt/malicious file",
                name, ndim_raw, MAX_REASONABLE_COUNT
            );
        }
        let ndim = ndim_raw as usize;
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
            // Only F32, F16, Q4_0, Q8_0 are supported today. Silently reinterpreting
            // an unrecognized GGML quant type as Q4_K would misread the raw tensor
            // bytes with the wrong block layout/scale format, corrupting weights
            // without any indication something went wrong — so we reject it instead.
            other => bail!(
                "Unsupported GGML tensor type {} for tensor {:?}: only F32(0), F16(1), Q4_0(2), Q8_0(8) are supported",
                other, name
            ),
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
    // `general.alignment: 0` (or a negative i32 that lands on 0) would divide-by-zero
    // in the padding computation below — reject it explicitly instead of panicking.
    if alignment == 0 {
        bail!("GGUF general.alignment is 0 — corrupt/malicious file (alignment must be nonzero)");
    }

    let header_size = r.pos;
    let padding = (alignment - (header_size % alignment)) % alignment;
    let data_section_start = header_size + padding;

    // Bounds-check every tensor's start offset AND declared byte length up
    // front — a corrupt/truncated GGUF file with an out-of-range offset or a
    // tensor whose (shape, dtype) claims more bytes than exist between its
    // offset and EOF must fail loudly here, not panic/read-out-of-bounds
    // later on first access via `tensor()`.
    let mut tensor_meta = HashMap::new();
    for t in temp_tensors {
        let tensor_start = data_section_start + t.offset as usize;
        if tensor_start > data_slice.len() {
            bail!(
                "GGUF tensor {:?} offset {} exceeds file length {}: file is truncated or corrupt",
                t.name, tensor_start, data_slice.len()
            );
        }

        let elem_count: usize = t.shape.iter().product();
        let byte_len = dtype_byte_len(t.dtype, elem_count)
            .with_context(|| format!("tensor {:?}", t.name))?;
        if tensor_start + byte_len > data_slice.len() {
            bail!(
                "GGUF tensor {:?} declares {} bytes (shape {:?}, dtype {:?}) starting at offset \
                 {}, but the file only has {} bytes remaining: file is truncated or corrupt",
                t.name, byte_len, t.shape, t.dtype, tensor_start, data_slice.len() - tensor_start
            );
        }

        // Silently overwriting a duplicate tensor name would leave `tensor_meta`
        // pointing at whichever of the two entries won the race, hiding a
        // malformed/ambiguous file instead of failing loudly.
        if tensor_meta.contains_key(&t.name) {
            bail!("GGUF file declares tensor {:?} more than once", t.name);
        }

        tensor_meta.insert(t.name, StoredTensorMeta {
            offset: tensor_start,
            len: byte_len,
            shape: t.shape,
            dtype: t.dtype,
        });
    }

    Ok(GgufFile {
        _mmap: mmap,
        metadata,
        tensor_meta,
    })
}
