use std::fs::File;
use std::io::Read;
use std::collections::HashMap;

use crate::gguf::header;
use crate::gguf::metadata;

pub struct GgufFile {
  pub header: header::GgufHeader,
  pub metadata: HashMap<String, metadata::GgufMetadataValue>,
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

  Ok(GgufFile {
    header: header,
    metadata: metadata,
  })
}