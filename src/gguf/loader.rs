use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use crate::gguf::header;
use crate::gguf::metadata;
use crate::gguf::tensors;

#[allow(dead_code)]
pub struct GgufFile {
  pub header: header::GgufHeader,
  pub metadata: HashMap<String, metadata::GgufMetadataValue>,
  pub tensors: HashMap<String, tensors::GgufTensor>,
}

pub fn load(path: &str) -> Result<GgufFile, String> {
  // Open the file
  let file = File::open(path)
    .map_err(|e| format!("Failed to open file: {}", e))?;
  let mut reader = std::io::BufReader::new(file);

  // Read the header
  let mut header_bytes = [0u8; 24];
  reader.read_exact(&mut header_bytes)
    .map_err(|e| format!("Failed to read header: {}", e))?;
  let header = header::parse(&header_bytes)?;

  // Read the metadata
  let metadata = metadata::parse(&mut reader, header.kv_count)?;

  // Read the tensors
  let tensors = tensors::parse(&mut reader, header.tensor_count)?;

  Ok(GgufFile {
    header: header,
    metadata: metadata,
    tensors: tensors,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bonsai_gguf() {
    let bonsai_model = load("./models/bonsai-1.7.gguf")
      .expect("Failed to load model");

    assert_eq!(bonsai_model.header.kv_count, 35);
    assert_eq!(bonsai_model.metadata.len(), 35);
    assert_eq!(bonsai_model.header.tensor_count, 310);
    assert_eq!(bonsai_model.tensors.len(), 310);
  }

  #[test]
  fn test_falcon_gguf() {
    let falcon_model = load("./models/falcon-3-1b.gguf")
      .expect("Failed to load model");

    assert_eq!(falcon_model.header.kv_count, 23);
    assert_eq!(falcon_model.metadata.len(), 23);
    assert_eq!(falcon_model.header.tensor_count, 165);
    assert_eq!(falcon_model.tensors.len(), 165);
  }
}