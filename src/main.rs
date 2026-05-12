mod gguf;
mod tokenizer;

fn main() {
  let bonsai_model = gguf::loader::load("./models/bonsai-1.7.gguf")
    .expect("Failed to load model");

  let tokenized_input = tokenizer::loader::tokenize(&bonsai_model, "This is a test message!")
    .expect("Failed to tokenize message");
}
