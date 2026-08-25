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
        # The CUDA toolkit is unfree, so this flake instantiates its own
        # nixpkgs with `allowUnfree` rather than relying on the caller's
        # config. Only the CUDA dev shell pulls unfree paths in; the default
        # package below never references them.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Pinned to 13.2 to match the `nvcc`/cuRAND pairing the kernels are
        # built against, and because Blackwell (sm_100/sm_120) needs 13.x.
        cudaPackages = pkgs.cudaPackages_13_2;

        # Only the pieces the kernels actually need: nvcc, the CUDA runtime
        # and CCCL headers it includes, and cuRAND. Merged into one tree so
        # CUDA_PATH resolves bin/ and include/ the way nvcc and cubecl-cuda
        # expect. Deliberately not `cudaPackages.cudatoolkit`, which drags in
        # cuda_gdb, Nsight and friends for no benefit here.
        cudaEnv = pkgs.symlinkJoin {
          name = "cuda-env-${cudaPackages.cudaMajorMinorVersion}";
          paths = with cudaPackages; [
            cuda_nvcc
            # CUDA 13 split the CRT headers (crt/host_config.h, which
            # cuda_runtime.h pulls in) into their own redistributable.
            cuda_crt
            cuda_cudart
            cccl
            libcurand
            libcurand.dev
            libcurand.include
            libcurand.lib
          ];
        };

        rustDevPackages = with pkgs; [
          cargo
          rustc
          clippy
          rustfmt
          rust-analyzer
        ];

        # `libcuda.so` ships with the NVIDIA *driver*, never with the
        # toolkit, so it cannot come from nixpkgs. NixOS exposes it at
        # /run/opengl-driver/lib; other distributions keep it on the system
        # loader path. This is the one unavoidable impurity for actually
        # running (not building) CUDA code.
        driverLibPath = "/run/opengl-driver/lib:/run/opengl-driver-32/lib";
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
          nativeBuildInputs = [ pkgs.makeWrapper ];
          postInstall = ''
            wrapProgram $out/bin/permanent \
              --prefix LD_LIBRARY_PATH : ${pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ]}
          '';
          # .cargo/config.toml sets target-cpu=native which is impure; the
          # source filter above drops it so the nix build stays reproducible.
        };

        devShells.default = pkgs.mkShell {
          packages = rustDevPackages ++ (with pkgs; [
            pkg-config
            vulkan-loader
          ]);
          env.RUST_BACKTRACE = "1";
          env.LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ];
          # `nix develop` is impure and inherits the caller's environment, so
          # an ambient CUDA_PATH (e.g. a system /opt/cuda) would leak in and
          # a `--features cuda` build here would silently link against it.
          # Drop it: CUDA belongs to the `.#cuda` shell, which pins its own.
          shellHook = ''
            unset CUDA_PATH NVCC_PREPEND_FLAGS
          '';
        };

        # `nix develop .#cuda` — adds a reproducible CUDA 13.2 toolchain on
        # top of the default shell. Everything (nvcc, cuRAND headers, the
        # host compiler nvcc drives) comes from nixpkgs; nothing is taken
        # from a system-wide /opt/cuda or /usr/local/cuda.
        devShells.cuda = pkgs.mkShell {
          # `backendStdenv` is the stdenv nixpkgs builds CUDA with: its gcc
          # is one nvcc 13.2 accepts, and it avoids the binutils/glibc
          # mismatch you get from pointing a system CUDA at a nix host
          # toolchain.
          stdenv = cudaPackages.backendStdenv;

          packages = rustDevPackages ++ (with pkgs; [
            pkg-config
            vulkan-loader
          ]) ++ [
            cudaEnv
            # Nsight Compute, for grounding kernel-level performance claims.
            cudaPackages.nsight_compute
          ];

          env.RUST_BACKTRACE = "1";
          # cubecl-cuda and our build.rs both look at CUDA_PATH first; point
          # it at the merged toolkit tree so bin/nvcc and include/ resolve.
          env.CUDA_PATH = "${cudaEnv}";
          # nvcc derives its default include dir from realpath(nvcc)/../include,
          # which resolves back into the cuda_nvcc store path alone and misses
          # the cudart/CRT/cuRAND headers. Point it at the merged tree.
          env.NVCC_PREPEND_FLAGS = "-I${cudaEnv}/include";
          env.LD_LIBRARY_PATH =
            "${pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader cudaEnv ]}:${driverLibPath}";
        };
      }
    );
}
