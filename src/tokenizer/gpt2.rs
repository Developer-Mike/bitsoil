use crate::gguf::loader::GgufFile;

pub fn tokenize(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  let tokens = file.metadata.get("tokenizer.ggml.tokens")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.tokens'".to_string())?
    .as_array().ok_or("Tokenizer metadata 'tokenizer.ggml.tokens' is not an array".to_string())?
    .iter().map(|v| v.as_string().ok_or("Tokenizer metadata 'tokenizer.ggml.tokens' contains non-string value".to_string()))
    .collect::<Result<Vec<&String>, String>>()?;
  let merges = file.metadata.get("tokenizer.ggml.merges")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.merges'".to_string())?
    .as_array().ok_or("Tokenizer metadata 'tokenizer.ggml.merges' is not an array".to_string())?
    .iter().map(|v| v.as_string().ok_or("Tokenizer metadata 'tokenizer.ggml.merges' contains non-string value".to_string()))
    .collect::<Result<Vec<&String>, String>>()?;
  let eos_token_id = file.metadata.get("tokenizer.ggml.eos_token_id")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.eos_token_id'".to_string())?
    .as_uint32().ok_or("Tokenizer metadata 'tokenizer.ggml.eos_token_id' is not a uint64".to_string())? as u32;

  Err("GPT-2 tokenizer is not implemented yet".to_string())
}