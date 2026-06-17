{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
  in
  {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        rustc cargo
        rust-analyzer
      ];

      shellHook = ''
        if [ ! -f ./models/bonsai-1.7b.gguf ]; then
          echo "Downloading bonsai-1.7b.gguf..."
          mkdir -p ./models
          curl -L -o ./models/bonsai-1.7b.gguf https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-F16.gguf
        fi
      '';
    };
  };
}
