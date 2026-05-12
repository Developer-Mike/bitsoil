use crate::gguf::loader::GgufFile;
use crate::gguf::metadata::GgufMetadataValue;

pub fn tokenize(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  /*
  tokenizer.ggml.tokens	Array[String]	The vocabulary
  tokenizer.ggml.scores	Array[Float32]	Token scores/log-probs
  tokenizer.ggml.merges	Array[String]	BPE merge pairs ("a b")
  tokenizer.ggml.bos_token_id	UInt32	Begin-of-sequence token ID
  */

  Err("GPT-2 tokenizer is not implemented yet".to_string())
}