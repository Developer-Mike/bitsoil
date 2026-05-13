use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::GgufMetadataValue;

mod gpt2;

pub fn tokenize(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  if !file.metadata.contains_key("tokenizer.ggml.model") {
    return Err("Model does not contain tokenizer metadata".to_string());
  }

  let model_gguf_value = file.metadata.get("tokenizer.ggml.model").unwrap();
  match model_gguf_value {
    GgufMetadataValue::String(model) => match model.as_str() {
      "gpt2" => return gpt2::tokenize(file, input),
      _ => return Err(format!("Unsupported tokenizer model: {}", model)),
    },
    _ => return Err(format!("Unsupported tokenizer model metadata type: {:?}", model_gguf_value)),
  }
}