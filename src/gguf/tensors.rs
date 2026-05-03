use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct GgufTensor {
  pub info: GgufTensorInfo,
  pub weights: Vec<u8>,
}

#[allow(dead_code)]
pub struct GgufTensorInfo {
  name: String,
  shape: Vec<u64>,
  quant_type: u32,
  offset: u64,
}

pub fn parse(reader: &mut BufReader<File>, tensors_count: u64) -> Result<HashMap<String, GgufTensor>, String> {
  let mut tensor_infos = Vec::with_capacity(tensors_count as usize);

  for _ in 0..tensors_count {
    let info = parse_info(reader)?;
    tensor_infos.push(info);
  }

  // Get next 32 byte boundary aligned offset
  let offset = reader.seek(SeekFrom::Current(0))
    .map_err(|e| format!("Failed to get current reader offset: {}", e))?;
  let weight_offset = offset + (32 - (offset % 32)) % 32;

  let mut tensors = HashMap::new();
  for mut info in tensor_infos {
    info.offset = weight_offset + info.offset;
    let weights = parse_weights(reader, &info)?;

    tensors.insert(info.name.clone(), GgufTensor {
      info: info,
      weights: weights,
    });
  }

  Ok(tensors)
}

fn parse_info(reader: &mut BufReader<File>) -> Result<GgufTensorInfo, String> {
  let mut name_len_bytes = [0u8; 8];
  reader.read_exact(&mut name_len_bytes)
    .map_err(|e| format!("Failed to read tensor name length: {}", e))?;
  let name_len = u64::from_le_bytes(name_len_bytes) as usize;

  let mut name_bytes = vec![0u8; name_len];
  reader.read_exact(&mut name_bytes)
    .map_err(|e| format!("Failed to read tensor name: {}", e))?;
  let name = String::from_utf8(name_bytes)
    .map_err(|e| format!("Failed to parse tensor name as UTF-8: {}", e))?;

  let mut dim_count_bytes = [0u8; 4];
  reader.read_exact(&mut dim_count_bytes)
    .map_err(|e| format!("Failed to read tensor dimension count: {}", e))?;
  let dim_count = u32::from_le_bytes(dim_count_bytes) as usize;

  let mut shape = Vec::with_capacity(dim_count);
  for _ in 0..dim_count {
    let mut dim_size_bytes = [0u8; 8];
    reader.read_exact(&mut dim_size_bytes)
      .map_err(|e| format!("Failed to read tensor dimension size: {}", e))?;
    let dim_size = u64::from_le_bytes(dim_size_bytes);

    shape.push(dim_size);
  }

  let mut quant_type_bytes = [0u8; 4];
  reader.read_exact(&mut quant_type_bytes)
    .map_err(|e| format!("Failed to read tensor quantization type: {}", e))?;
  let quant_type = u32::from_le_bytes(quant_type_bytes);

  let mut offset_bytes = [0u8; 8];
  reader.read_exact(&mut offset_bytes)
    .map_err(|e| format!("Failed to read tensor data offset: {}", e))?;
  let offset = u64::from_le_bytes(offset_bytes);

  Ok(GgufTensorInfo {
    name: name,
    shape: shape,
    quant_type: quant_type,
    offset: offset,
  })
}

fn parse_weights(reader: &mut BufReader<File>, info: &GgufTensorInfo) -> Result<Vec<u8>, String> {
  let num_elements: u64 = info.shape.iter().product();
  let element_size: u64 = get_weight_entry_size(info.quant_type)? as u64;
  let total_size = (num_elements * element_size) as usize;

  let mut weights = vec![0u8; total_size];
  reader.seek(SeekFrom::Start(info.offset))
    .map_err(|e| format!("Failed to seek to tensor weights: {}", e))?;
  reader.read_exact(&mut weights)
    .map_err(|e| format!("Failed to read tensor weights: {}", e))?;

  Ok(weights)
}

// TODO: Add support for other quant types
// https://github.com/ggml-org/llama.cpp/blob/d05fe1d7dadbf8943c8f1903fcf65b935ddab839/gguf-py/gguf/constants.py#L3993
fn get_weight_entry_size(quant_type: u32) -> Result<f32, String> {
  match quant_type {
    0 => Ok(4.0), // f32
    42 => Ok(0.5), // Ternary quantization (2 bits per element)
    _ => Err(format!("Unknown quantization type: {}", quant_type)),
  }
}