{
  description = "Approximate permanent computation via simulated annealing on matchings";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "permanent";
          version = "0.1.0";
          src = pkgs.lib.cleanSourceWith {
            src = self;
            filter =
              path: type:
              let
                base = baseNameOf path;
              in
              base != "flake.nix" && base != "flake.lock" && base != ".cargo";
          };
          cargoLock.lockFile = ./Cargo.lock;
          # .cargo/config.toml sets target-cpu=native which is impure; the
          # source filter above drops it so the nix build stays reproducible.
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer
          ];
          env.RUST_BACKTRACE = "1";
        };
      }
    );
}
