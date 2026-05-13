use regex::Regex;

use crate::gguf::loader::GgufFile;

pub fn tokenize(file: &GgufFile, input: &str) -> Result<Vec<u32>, String> {
  let tokens = file.metadata.get("tokenizer.ggml.tokens")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.tokens'".to_string())?
    .as_array().ok_or("Tokenizer metadata 'tokenizer.ggml.tokens' is not an array".to_string())?
    .iter().map(|v| v.as_string().ok_or("Tokenizer metadata 'tokenizer.ggml.tokens' contains non-string value".to_string()))
    .collect::<Result<Vec<&str>, String>>()?;
  let merges = file.metadata.get("tokenizer.ggml.merges")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.merges'".to_string())?
    .as_array().ok_or("Tokenizer metadata 'tokenizer.ggml.merges' is not an array".to_string())?
    .iter().map(|v| v.as_string().ok_or("Tokenizer metadata 'tokenizer.ggml.merges' contains non-string value".to_string()))
    .collect::<Result<Vec<&str>, String>>()?;
  let eos_token_id = file.metadata.get("tokenizer.ggml.eos_token_id")
    .ok_or("Tokenizer metadata is missing 'tokenizer.ggml.eos_token_id'".to_string())?
    .as_uint32().ok_or("Tokenizer metadata 'tokenizer.ggml.eos_token_id' is not a uint64".to_string())? as u32;

  let pre_tokenized_input = pre_tokenize(input);
  let merged_pairs = merge_pairs(pre_tokenized_input, &merges);
  let token_ids = lookup_tokens(&merged_pairs, &tokens);

  Ok(token_ids)
}

fn pre_tokenize(input: &str) -> Vec<&str> {
  let pre_tokenize_regex = Regex::new(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+").unwrap();
  let matches = pre_tokenize_regex.find_iter(input);

  matches.map(|m: regex::Match| m.as_str()).collect()
}

fn merge_pairs(tokens: Vec<&str>, merges: &Vec<&str>) -> Vec<u32> {
  vec![]
}

fn lookup_tokens(pairs: &Vec<u32>, tokens: &Vec<&str>) -> Vec<u32> {
  // TODO: HashMap/Set
  pairs.to_vec()
}