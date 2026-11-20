{
  description = "Permanent of a nonnegative matrix to a requested accuracy, by population annealing";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # The cuda backend needs the exact toolchain of rust-toolchain.toml, with
    # rustc-dev, which nixpkgs' rustc does not ship.
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    # Run-time discovery of the host NVIDIA driver libraries on non-NixOS
    # hosts (see driverLibPath below for why this cannot come from nixpkgs).
    nix-gl-host.url = "github:numtide/nix-gl-host";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      nix-gl-host,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # The CUDA toolkit is unfree, so this flake instantiates its own
        # nixpkgs with `allowUnfree`. Only the CUDA dev shell pulls unfree
        # paths in; the packages below never reference them.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ rust-overlay.overlays.default ];
        };

        # cuda-oxide wants CUDA 13 and a 580-series driver.
        cudaPackages = pkgs.cudaPackages_13;

        # What cuda-oxide links device code against: libNVVM and libdevice
        # (shipped with nvcc), nvJitLink, and the runtime headers. Merged
        # into one tree so CUDA_PATH resolves the way the toolkit lays it out.
        cudaEnv = pkgs.symlinkJoin {
          name = "cuda-env-${cudaPackages.cudaMajorMinorVersion}";
          paths = with cudaPackages; [
            cuda_nvcc
            cuda_crt
            cuda_cudart
            cuda_nvrtc
            libnvjitlink
            # cuda-oxide's driver bindings include curand.h, which nixpkgs
            # keeps in an output of its own
            libcurand
            libcurand.dev
            libcurand.include
            libcurand.lib
          ];
        };

        # cuda-oxide generates its driver bindings with bindgen.
        libclang = pkgs.llvmPackages_21.libclang;

        cudaToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        rustDevPackages = with pkgs; [
          cargo
          rustc
          clippy
          rustfmt
          rust-analyzer
        ];

        # `libcuda.so` ships with the NVIDIA *driver*, never with the
        # toolkit, so it cannot come from nixpkgs. NixOS exposes it at
        # /run/opengl-driver/lib. On FHS distributions the driver libs sit
        # next to the system glibc, which must never land on a nix binary's
        # LD_LIBRARY_PATH wholesale; nixglhost copies just the driver DSOs
        # into an isolated cache directory and prints that path.
        driverLibPath = "/run/opengl-driver/lib";
        nixglhost = nix-gl-host.packages.${system}.default;

        # The program with the rayon backend. The cuda backend is built with
        # cuda-oxide inside the cuda shell.
        permanent =
          { withMpi }:
          pkgs.rustPlatform.buildRustPackage {
            pname = "permanent" + pkgs.lib.optionalString withMpi "-mpi";
            version = (pkgs.lib.importTOML ./Cargo.toml).workspace.package.version;
            src = pkgs.lib.cleanSourceWith {
              src = self;
              filter =
                path: type:
                let
                  base = baseNameOf path;
                in
                # .cargo/config.toml sets target-cpu=native, which is impure
                !(builtins.elem base [
                  "flake.nix"
                  "flake.lock"
                  ".cargo"
                  "rust-toolchain.toml"
                  "misc"
                ]);
            };
            cargoLock = {
              lockFile = ./Cargo.lock;
              # the optional cuda backend brings git dependencies into the lock
              allowBuiltinFetchGit = true;
            };
            cargoBuildFlags = [ "--package" "permanent-cli" ];
            buildFeatures = pkgs.lib.optional withMpi "mpi";
            # the MPI bindings are generated with bindgen against mpi.h
            nativeBuildInputs = pkgs.lib.optionals withMpi [
              pkgs.rustPlatform.bindgenHook
              pkgs.mpi
            ];
            buildInputs = pkgs.lib.optional withMpi pkgs.mpi;
            meta.mainProgram = "permanent";
          };
      in
      {
        packages.default = permanent { withMpi = false; };
        # `mpirun -n R permanent ...`
        packages.mpi = permanent { withMpi = true; };

        devShells.default = pkgs.mkShell {
          packages = rustDevPackages;
          env.RUST_BACKTRACE = "1";
        };

        # `nix develop .#mpi` — for `cargo build --features mpi`.
        devShells.mpi = pkgs.mkShell {
          packages = rustDevPackages ++ [ pkgs.mpi ];
          nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook ];
          env.RUST_BACKTRACE = "1";
        };

        # `nix develop .#cuda` — for the cuda backend: `cargo oxide build --
        # --profile cuda --features cuda`. Everything but the driver comes from
        # nixpkgs. cargo-oxide itself is not packaged: it fetches and builds
        # its compiler backend on first use, so install it once inside this
        # shell with the command the shell prints.
        devShells.cuda = pkgs.mkShell {
          packages = [
            cudaToolchain
            cudaEnv
            libclang
            pkgs.mpi
            nixglhost
          ];
          nativeBuildInputs = [ pkgs.rustPlatform.bindgenHook ];

          env.RUST_BACKTRACE = "1";
          env.CUDA_PATH = "${cudaEnv}";
          env.CUDA_HOME = "${cudaEnv}";
          env.LD_LIBRARY_PATH = "${pkgs.lib.makeLibraryPath [ cudaEnv ]}:${driverLibPath}";
          shellHook = ''
            # keep cargo-installed tools of this shell apart from the user's
            export CARGO_HOME="''${CARGO_HOME:-$HOME/.cargo}"
            export PATH="$CARGO_HOME/bin:$PATH"
            # Where the driver lives somewhere nixglhost does not look (a
            # container with /usr/lib64-nvidia, say), add that directory to
            # LD_LIBRARY_PATH by hand.
            if [ ! -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(nixglhost -p 2>/dev/null || true)"
            fi
            if ! command -v cargo-oxide >/dev/null; then
              echo "cargo-oxide is not installed; run:"
              echo "  cargo install --locked --git https://github.com/NVIDIA/cuda-rust.git cargo-oxide"
            fi
          '';
        };
      }
    );
}
