use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::GgufMetadataValue;

pub fn tokenize(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  if !file.metadata.contains_key("tokenizer.ggml.model") {
    return Err("Model does not contain tokenizer metadata".to_string());
  }

  let model_gguf_value = file.metadata.get("tokenizer.ggml.model").unwrap();
  match model_gguf_value {
    GgufMetadataValue::String(model) => match model.as_str() {
      "gpt2" => return tokenize_gpt2(file, input),
      _ => return Err(format!("Unsupported tokenizer model: {}", model)),
    },
    _ => return Err(format!("Unsupported tokenizer model metadata type: {:?}", model_gguf_value)),
  }

  Err("Tokenizer model metadata is not a string".to_string())
}

fn tokenize_gpt2(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  Err("GPT-2 tokenizer is not implemented yet".to_string())
}