mod gguf;

fn main() {
  let model = gguf::loader::load("./models/falcon-3-1b.gguf") // ./models/bonsai-1.7.gguf
    .expect("Failed to load model");

  dbg!(model.metadata.len());
  dbg!(model.tensors.len());
}
