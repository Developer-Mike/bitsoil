#[allow(dead_code)]
pub struct GgufHeader {
  pub version: u32,
  pub kv_count: u64,
  pub tensor_count: u64,
}

pub fn parse(header: &[u8; 24]) -> Result<GgufHeader, String> {
  let magic = &header[0..4];

  if magic != b"GGUF" {
    return Err(String::from("Invalid magic number"));
  }

  let version = u32::from_le_bytes(
    header[4..8].try_into()
      .map_err(|_| String::from("Failed to parse version"))?,
  );
  let tensor_count = u64::from_le_bytes(
    header[8..16].try_into()
      .map_err(|_| String::from("Failed to parse key-value count"))?,
  );
  let kv_count = u64::from_le_bytes(
    header[16..24].try_into()
        .map_err(|_| String::from("Failed to parse tensor count"))?,
  );

  Ok(GgufHeader {
    version: version,
    kv_count: kv_count,
    tensor_count: tensor_count,
  })
}