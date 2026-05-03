mod gguf;

fn main() {
  let model = gguf::loader::load("./models/bonsai-1.7.gguf") // ./models/falcon-3-1b.gguf
    .expect("Failed to load model");

  dbg!(model.metadata.len());
  dbg!(model.tensors.len());
}
