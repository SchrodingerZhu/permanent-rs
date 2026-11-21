// Loaded by Deno before the wasm-bindgen test glue (see flake.nix).

// The glue assigns to `window`, which Deno 2 no longer defines.
globalThis.window = globalThis;

// wgpu reads every limit of the WebGPU specification; these are the ones
// Deno's adapter does not report yet, with the specification's defaults.
const defaults = { maxBindGroupsPlusVertexBuffers: 24, maxInterStageShaderVariables: 16 };
for (const [name, value] of Object.entries(defaults)) {
  if (!(name in GPUSupportedLimits.prototype)) {
    Object.defineProperty(GPUSupportedLimits.prototype, name, { get: () => value });
  }
}
