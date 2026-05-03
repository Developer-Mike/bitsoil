use std::fs::File;
use std::io::{BufReader, Read};
use std::collections::HashMap;

#[allow(dead_code)]
pub enum GgufMetadataValue {
  UInt8(u8),
  Int8(i8),
  UInt16(u16),
  Int16(i16),
  UInt32(u32),
  Int32(i32),
  Float32(f32),
  Bool(bool),
  String(String),
  Array(Vec<GgufMetadataValue>),
  UInt64(u64),
  Int64(i64),
  Float64(f64),
}

pub fn parse(reader: &mut BufReader<File>, kv_count: u64) -> Result<HashMap<String, GgufMetadataValue>, String> {
  let mut metadata = HashMap::new();

  for _ in 0..kv_count {
    // Read the key length
    let mut key_len_bytes = [0u8; 8];
    reader.read_exact(&mut key_len_bytes)
      .map_err(|e| format!("Failed to read key length: {}", e))?;
    let key_len = u64::from_le_bytes(key_len_bytes) as usize;

    // Read the key
    let mut key_bytes = vec![0u8; key_len];
    reader.read_exact(&mut key_bytes)
      .map_err(|e| format!("Failed to read key: {}", e))?;
    let key = String::from_utf8(key_bytes)
      .map_err(|e| format!("Failed to parse key as UTF-8: {}", e))?;

    // Read the value type
    let mut value_type_byte = [0u8; 4];
    reader.read_exact(&mut value_type_byte)
      .map_err(|e| format!("Failed to read value type: {}", e))?;
    let value_type = u32::from_le_bytes(value_type_byte);

    // Read the value
    let value = parse_value(reader, value_type)?;

    metadata.insert(key, value);
  }

  Ok(metadata)
}

fn parse_value(reader: &mut BufReader<File>, value_type: u32) -> Result<GgufMetadataValue, String> {
  match value_type {
    0 => Ok(GgufMetadataValue::UInt8(parse_uint8(reader)?)),
    1 => Ok(GgufMetadataValue::Int8(parse_int8(reader)?)),
    2 => Ok(GgufMetadataValue::UInt16(parse_uint16(reader)?)),
    3 => Ok(GgufMetadataValue::Int16(parse_int16(reader)?)),
    4 => Ok(GgufMetadataValue::UInt32(parse_uint32(reader)?)),
    5 => Ok(GgufMetadataValue::Int32(parse_int32(reader)?)),
    6 => Ok(GgufMetadataValue::Float32(parse_float32(reader)?)),
    7 => Ok(GgufMetadataValue::Bool(parse_bool(reader)?)),
    8 => Ok(GgufMetadataValue::String(parse_string(reader)?)),
    9 => Ok(GgufMetadataValue::Array(parse_array(reader)?)),
    10 => Ok(GgufMetadataValue::UInt64(parse_uint64(reader)?)),
    11 => Ok(GgufMetadataValue::Int64(parse_int64(reader)?)),
    12 => Ok(GgufMetadataValue::Float64(parse_float64(reader)?)),
    default => Err(format!("Unsupported value type: {}", default)),
  }
}

fn parse_uint8(reader: &mut BufReader<File>) -> Result<u8, String> {
  let mut val_bytes = [0u8; 1];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read uint8 value: {}", e))?;

  Ok(val_bytes[0])
}

fn parse_int8(reader: &mut BufReader<File>) -> Result<i8, String> {
  let mut val_bytes = [0u8; 1];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read int8 value: {}", e))?;

  Ok(val_bytes[0] as i8)
}

fn parse_uint16(reader: &mut BufReader<File>) -> Result<u16, String> {
  let mut val_bytes = [0u8; 2];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read uint16 value: {}", e))?;

  Ok(u16::from_le_bytes(val_bytes))
}

fn parse_int16(reader: &mut BufReader<File>) -> Result<i16, String> {
  let mut val_bytes = [0u8; 2];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read int16 value: {}", e))?;

  Ok(i16::from_le_bytes(val_bytes))
}

fn parse_uint32(reader: &mut BufReader<File>) -> Result<u32, String> {
  let mut val_bytes = [0u8; 4];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read uint32 value: {}", e))?;

  Ok(u32::from_le_bytes(val_bytes))
}

fn parse_int32(reader: &mut BufReader<File>) -> Result<i32, String> {
  let mut val_bytes = [0u8; 4];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read int32 value: {}", e))?;

  Ok(i32::from_le_bytes(val_bytes))
}

fn parse_float32(reader: &mut BufReader<File>) -> Result<f32, String> {
  let mut val_bytes = [0u8; 4];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read float32 value: {}", e))?;

  Ok(f32::from_le_bytes(val_bytes))
}

fn parse_bool(reader: &mut BufReader<File>) -> Result<bool, String> {
  let mut val_bytes = [0u8; 1];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read bool value: {}", e))?;

  match val_bytes[0] {
    0 => Ok(false),
    1 => Ok(true),
    other => Err(format!("Invalid bool value: {}", other)),
  }
}

fn parse_string(reader: &mut BufReader<File>) -> Result<String, String> {
  let mut len_bytes = [0u8; 8];
  reader.read_exact(&mut len_bytes)
    .map_err(|e| format!("Failed to read string length: {}", e))?;
  let len = u64::from_le_bytes(len_bytes) as usize;

  let mut str_bytes = vec![0u8; len];
  reader.read_exact(&mut str_bytes)
    .map_err(|e| format!("Failed to read string value: {}", e))?;

  String::from_utf8(str_bytes)
    .map_err(|e| format!("Failed to parse string as UTF-8: {}", e))
}

fn parse_array(reader: &mut BufReader<File>) -> Result<Vec<GgufMetadataValue>, String> {
  let mut type_bytes = [0u8; 4];
  reader.read_exact(&mut type_bytes)
    .map_err(|e| format!("Failed to read array element type: {}", e))?;
  let element_type = u32::from_le_bytes(type_bytes);

  let mut len_bytes = [0u8; 8];
  reader.read_exact(&mut len_bytes)
    .map_err(|e| format!("Failed to read array length: {}", e))?;
  let len = u64::from_le_bytes(len_bytes);

  let mut arr_values = Vec::new();
  for _ in 0..len {
    let value = parse_value(reader, element_type)?;
    arr_values.push(value);
  }

  Ok(arr_values)
}

fn parse_uint64(reader: &mut BufReader<File>) -> Result<u64, String> {
  let mut val_bytes = [0u8; 8];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read uint64 value: {}", e))?;

  Ok(u64::from_le_bytes(val_bytes))
}

fn parse_int64(reader: &mut BufReader<File>) -> Result<i64, String> {
  let mut val_bytes = [0u8; 8];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read int64 value: {}", e))?;

  Ok(i64::from_le_bytes(val_bytes))
}

fn parse_float64(reader: &mut BufReader<File>) -> Result<f64, String> {
  let mut val_bytes = [0u8; 8];
  reader.read_exact(&mut val_bytes)
    .map_err(|e| format!("Failed to read float64 value: {}", e))?;

  Ok(f64::from_le_bytes(val_bytes))
}