mod gguf;
mod tokenizer;

fn main() {
  let bonsai_model = gguf::loader::load("./models/bonsai-1.7b.gguf")
    .expect("Failed to load model");

  println!("Loaded model: {:?}", bonsai_model.metadata);

  let tokenized_input = tokenizer::tokenize(&bonsai_model, "This is a test message!")
    .expect("Failed to tokenize message");

  println!("Tokenized input: {:?}", tokenized_input);
}
