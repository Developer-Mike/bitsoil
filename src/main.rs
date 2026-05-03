mod gguf;

fn main() {
  let bonsai_model = gguf::loader::load("./models/bonsai-1.7.gguf")
    .expect("Failed to load model");

  dbg!(bonsai_model.header.kv_count);
  dbg!(bonsai_model.metadata.len());
  dbg!(bonsai_model.header.tensor_count);
  dbg!(bonsai_model.tensors.len());

  let falcon_model = gguf::loader::load("./models/falcon-3-1b.gguf")
    .expect("Failed to load model");

  dbg!(falcon_model.header.kv_count);
  dbg!(falcon_model.metadata.len());
  dbg!(falcon_model.header.tensor_count);
  dbg!(falcon_model.tensors.len());
}
