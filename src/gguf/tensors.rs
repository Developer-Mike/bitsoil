use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::collections::HashMap;

#[allow(dead_code)]
pub struct GgufTensor {
  pub info: GgufTensorInfo,
  pub weights: GgufTensorWeights
}

#[allow(dead_code)]
pub struct GgufTensorInfo {
  pub name: String,
  pub shape: Vec<u64>,
  pub quant_type: GgufQuantType,
  pub offset: u64,
}

// TODO: Add support for other quant types
// https://github.com/ggml-org/llama.cpp/blob/d05fe1d7dadbf8943c8f1903fcf65b935ddab839/gguf-py/gguf/constants.py#L3993
#[allow(dead_code)]
#[derive(Debug)]
pub enum GgufQuantType {
  F32,
  TernaryBonsai,
}

impl From<u32> for GgufQuantType {
  fn from(value: u32) -> Self {
    match value {
      0 => GgufQuantType::F32,
      42 => GgufQuantType::TernaryBonsai,
      _ => panic!("Unknown quantization type: {}", value),
    }
  }
}

#[allow(dead_code)]
pub enum GgufTensorWeights {
  F16(Vec<f32>),
  F32(Vec<f32>),
  Ternary(Vec<i8>)
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
  let quant_type = GgufQuantType::from(u32::from_le_bytes(quant_type_bytes));

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

fn parse_weights(reader: &mut BufReader<File>, info: &GgufTensorInfo) -> Result<GgufTensorWeights, String> {
  let num_elements: usize = info.shape.iter().product::<u64>() as usize;
  let total_size: usize = match info.quant_type {
    GgufQuantType::F32 => num_elements * 4,
    GgufQuantType::TernaryBonsai => (num_elements / 64) * 13,
  };

  let mut weight_bytes = vec![0u8; total_size];
  reader.seek(SeekFrom::Start(info.offset))
    .map_err(|e| format!("Failed to seek to tensor weights: {}", e))?;
  reader.read_exact(&mut weight_bytes)
    .map_err(|e| format!("Failed to read tensor weights: {}", e))?;

  Ok(match info.quant_type {
    GgufQuantType::F32 => {
      let weights = weight_bytes.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
      GgufTensorWeights::F32(weights)
    }
    GgufQuantType::TernaryBonsai => {
      let num_blocks = num_elements / 64;
      let mut weights = Vec::with_capacity(num_elements);

      for block_i in 0..num_blocks {
        let offset = block_i * 13;
        let block_weights = &weight_bytes[offset..=offset + 12];
        let max_byte_i = block_weights.len() - 1;

        for byte_i in 0..max_byte_i {
          let byte = block_weights[byte_i];
          let last_byte = byte_i == max_byte_i;

          let combined = ((byte as u16) * 243) / 256;
          let base3 = combined % 81;

          weights.push((base3 % 3) as i8 - 1);
          weights.push(((base3 / 3) % 3) as i8 - 1);
          weights.push(((base3 / 9) % 3) as i8 - 1);
          weights.push((base3 / 27) as i8 - 1);

          if last_byte { continue; }
          weights.push((((byte as u16) * 3) >> 8) as i8 - 1);
        }
      }

      GgufTensorWeights::Ternary(weights)
    }
  })
}