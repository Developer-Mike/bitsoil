mod gguf;

fn main() {
  let bonsai_model = gguf::loader::load("./models/bonsai-1.7.gguf")
    .expect("Failed to load model");

  for (key, tensor) in &bonsai_model.tensors {
    println!("{key}: {:?}", tensor.info.quant_type);
  }
}
