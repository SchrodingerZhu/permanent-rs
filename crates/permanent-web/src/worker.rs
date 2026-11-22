//! Asynchronous typed client communicating with the Dedicated WebGPU Worker
//! (`assets/js/gpu-worker.js`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use futures_channel::oneshot;
use js_sys::{JSON, Reflect};
use permanent_demo::types::{GeneratedProblem, ProblemSpec, SampleSnapshot};
use permanent_wasm::types::{Capabilities, RunSummary, WebEvent, WebRunConfig};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{MessageEvent, Worker, WorkerOptions, WorkerType};

type PendingSender = oneshot::Sender<Result<Value, String>>;
type EventSink = Box<dyn FnMut(WebEvent)>;

struct Inner {
    worker: Worker,
    next_id: u32,
    pending: HashMap<u32, PendingSender>,
    on_event: Option<EventSink>,
    _onmessage: Closure<dyn FnMut(MessageEvent)>,
    _onerror: Closure<dyn FnMut(JsValue)>,
}

/// Cloneable handle to the single background WebGPU worker.
#[derive(Clone)]
pub struct GpuWorkerClient {
    inner: Rc<RefCell<Inner>>,
}

impl GpuWorkerClient {
    pub fn spawn() -> Result<Self, String> {
        let opts = WorkerOptions::new();
        opts.set_type(WorkerType::Module);
        let worker = Worker::new_with_options("./assets/js/gpu-worker.js", &opts)
            .map_err(|e| format!("failed to spawn WebGPU worker: {e:?}"))?;

        let pending: Rc<RefCell<Option<Rc<RefCell<Inner>>>>> = Rc::new(RefCell::new(None));
        let slot_msg = Rc::clone(&pending);
        let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |ev: MessageEvent| {
            let Some(inner_rc) = slot_msg.borrow().as_ref().cloned() else { return };
            let data = ev.data();
            let Ok(text_js) = JSON::stringify(&data) else { return };
            let Some(text) = text_js.as_string() else { return };
            let Ok(val) = serde_json::from_str::<Value>(&text) else { return };

            if val.get("type").and_then(Value::as_str) == Some("event") {
                if let Some(ev_val) = val.get("event").cloned()
                    && let Ok(web_event) = serde_json::from_value::<WebEvent>(ev_val)
                {
                    let mut inner = inner_rc.borrow_mut();
                    if let Some(cb) = inner.on_event.as_mut() {
                        cb(web_event);
                    }
                }
                return;
            }

            if let Some(id) = val.get("id").and_then(Value::as_u64).map(|n| n as u32) {
                let tx = inner_rc.borrow_mut().pending.remove(&id);
                if let Some(tx) = tx {
                    let ok = val.get("ok").and_then(Value::as_bool).unwrap_or(false);
                    if ok {
                        let payload = val.get("data").cloned().unwrap_or(Value::Null);
                        let _ = tx.send(Ok(payload));
                    } else {
                        let msg =
                            val.get("error").and_then(Value::as_str).unwrap_or("worker operation failed").to_owned();
                        let _ = tx.send(Err(msg));
                    }
                }
            }
        });

        let slot_err = Rc::clone(&pending);
        let onerror = Closure::<dyn FnMut(JsValue)>::new(move |err: JsValue| {
            let Some(inner_rc) = slot_err.borrow().as_ref().cloned() else { return };
            let message = Reflect::get(&err, &JsValue::from_str("message"))
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_else(|| "WebGPU worker error".into());
            let mut inner = inner_rc.borrow_mut();
            let waiters: Vec<_> = inner.pending.drain().collect();
            for (_, tx) in waiters {
                let _ = tx.send(Err(message.clone()));
            }
        });

        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        let inner = Rc::new(RefCell::new(Inner {
            worker,
            next_id: 1,
            pending: HashMap::new(),
            on_event: None,
            _onmessage: onmessage,
            _onerror: onerror,
        }));
        *pending.borrow_mut() = Some(Rc::clone(&inner));
        Ok(Self { inner })
    }

    async fn request<P: Serialize, R: DeserializeOwned>(&self, payload: P) -> Result<R, String> {
        let (id, rx) = {
            let mut inner = self.inner.borrow_mut();
            let id = inner.next_id;
            inner.next_id = inner.next_id.wrapping_add(1).max(1);
            let (tx, rx) = oneshot::channel();
            inner.pending.insert(id, tx);

            let mut obj = serde_json::to_value(payload).map_err(|e| e.to_string())?;
            if let Some(map) = obj.as_object_mut() {
                map.insert("id".into(), Value::from(id));
            }
            let text = serde_json::to_string(&obj).map_err(|e| e.to_string())?;
            let js_msg = JSON::parse(&text).map_err(|e| format!("{e:?}"))?;
            inner.worker.post_message(&js_msg).map_err(|e| format!("postMessage failed: {e:?}"))?;
            (id, rx)
        };

        let result = rx.await.map_err(|_| format!("worker channel closed (request {id})"))??;
        serde_json::from_value(result).map_err(|e| format!("decode worker response: {e}"))
    }

    pub async fn probe(&self) -> Result<Capabilities, String> {
        self.request(serde_json::json!({ "op": "probe" })).await
    }

    pub async fn build_problem(&self, spec: &ProblemSpec) -> Result<GeneratedProblem, String> {
        self.request(serde_json::json!({ "op": "build", "spec": spec })).await
    }

    pub async fn sample_covering(
        &self,
        spec: &ProblemSpec,
        seed: u64,
        steps: u32,
    ) -> Result<Option<SampleSnapshot>, String> {
        self.request(serde_json::json!({
            "op": "sample",
            "spec": spec,
            "seed": seed as f64,
            "steps": steps,
        }))
        .await
    }

    pub async fn run_estimation<F: FnMut(WebEvent) + 'static>(
        &self,
        spec: &ProblemSpec,
        config: &WebRunConfig,
        on_event: F,
    ) -> Result<Option<RunSummary>, String> {
        self.inner.borrow_mut().on_event = Some(Box::new(on_event));
        let res = self.request(serde_json::json!({ "op": "run", "spec": spec, "config": config })).await;
        self.inner.borrow_mut().on_event = None;
        res
    }

    pub fn cancel(&self) {
        if let Ok(js_msg) = JSON::parse(r#"{"op":"cancel"}"#) {
            let _ = self.inner.borrow().worker.post_message(&js_msg);
        }
    }
}
