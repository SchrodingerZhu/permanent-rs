// Dedicated ES-module Web Worker driving permanent-wasm (WebGPU WGSL engine).
// Keeps matrix scaling, covering sampling on large grids, and WebGPU runs
// completely off the Dioxus UI and Three.js main thread.

import init, {
  Problem,
  RunHandle,
  probe_capabilities,
} from "../../pkg-engine/permanent_wasm.js";

let readyPromise = null;
let keptProblem = null;
let runHandle = null;

async function ensureReady() {
  if (!readyPromise) {
    readyPromise = init().then(() => {
      runHandle = new RunHandle();
    });
  }
  await readyPromise;
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  const { id, op } = msg;

  // Immediate cooperative cancel signal (does not wait for a response id).
  if (op === "cancel") {
    if (runHandle) {
      runHandle.cancel();
    }
    return;
  }

  try {
    await ensureReady();

    if (op === "probe") {
      const caps = await probe_capabilities();
      self.postMessage({ id, ok: true, data: caps });
      return;
    }

    if (op === "build") {
      if (keptProblem) {
        keptProblem.free();
        keptProblem = null;
      }
      keptProblem = new Problem(msg.spec);
      const info = keptProblem.info();
      self.postMessage({ id, ok: true, data: info });
      return;
    }

    if (op === "sample") {
      if (!keptProblem) {
        if (!msg.spec) {
          throw new Error("no built problem in worker");
        }
        keptProblem = new Problem(msg.spec);
      }
      const snapshot = keptProblem.sample_covering(msg.seed, msg.steps);
      self.postMessage({ id, ok: true, data: snapshot });
      return;
    }

    if (op === "run") {
      const summary = await runHandle.run(msg.spec, msg.config, (ev) => {
        self.postMessage({ type: "event", event: ev });
      });
      self.postMessage({ id, ok: true, data: summary });
      return;
    }

    throw new Error(`unknown worker operation: ${op}`);
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      error: err && err.message ? err.message : String(err),
    });
  }
};
