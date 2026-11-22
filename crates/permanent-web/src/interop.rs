//! Clean JS interop helpers for theme management, hero canvas, Three.js 3D
//! viewport, clipboard copy, and JSON export.

use js_sys::{Array, Function, JSON, Reflect};
use permanent_demo::types::ProblemGeometry;
use serde::Serialize;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Blob, BlobPropertyBag, HtmlAnchorElement, Url, Window};

use crate::state::{ResolvedTheme, ThemeMode};

fn window() -> Option<Window> {
    web_sys::window()
}

pub fn load_theme_mode() -> ThemeMode {
    window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("permanent-theme").ok().flatten())
        .map(|v| ThemeMode::from_str(&v))
        .unwrap_or(ThemeMode::System)
}

pub fn system_prefers_dark() -> bool {
    window()
        .and_then(|w| w.match_media("(prefers-color-scheme: dark)").ok().flatten())
        .map(|mq| mq.matches())
        .unwrap_or(false)
}

pub fn resolve_theme(mode: ThemeMode) -> ResolvedTheme {
    match mode {
        ThemeMode::Light => ResolvedTheme::Light,
        ThemeMode::Dark => ResolvedTheme::Dark,
        ThemeMode::System => {
            if system_prefers_dark() {
                ResolvedTheme::Dark
            } else {
                ResolvedTheme::Light
            }
        }
    }
}

pub fn apply_theme_to_dom(mode: ThemeMode, resolved: ResolvedTheme) {
    if let Some(w) = window() {
        if let Ok(Some(storage)) = w.local_storage() {
            let _ = storage.set_item("permanent-theme", mode.as_str());
        }
        if let Some(el) = w.document().and_then(|d| d.document_element()) {
            let _ = el.set_attribute("data-theme", resolved.as_str());
            let _ = el.set_attribute("data-theme-mode", mode.as_str());
        }
    }
    hero_set_theme(resolved);
    three_set_theme(resolved);
}

fn call_global_method(namespace: &str, method: &str, args: &[JsValue]) {
    let Some(w) = window() else { return };
    let Ok(ns) = Reflect::get(&w, &JsValue::from_str(namespace)) else { return };
    if ns.is_undefined() || ns.is_null() {
        return;
    }
    let Ok(func) = Reflect::get(&ns, &JsValue::from_str(method)) else { return };
    let Ok(func) = func.dyn_into::<Function>() else { return };
    let js_args = Array::new();
    for arg in args {
        js_args.push(arg);
    }
    let _ = Reflect::apply(&func, &ns, &js_args);
}

fn to_js<T: Serialize>(value: &T) -> Option<JsValue> {
    let text = serde_json::to_string(value).ok()?;
    JSON::parse(&text).ok()
}

pub fn hero_mount(canvas_id: &str, theme: ResolvedTheme) {
    let Some(el) = window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(canvas_id)) else {
        return;
    };
    call_global_method("PermanentHero", "mount", &[el.into(), JsValue::from_str(theme.as_str())]);
}

pub fn hero_set_theme(theme: ResolvedTheme) {
    call_global_method("PermanentHero", "setTheme", &[JsValue::from_str(theme.as_str())]);
}

pub fn hero_dispose() {
    call_global_method("PermanentHero", "dispose", &[]);
}

pub fn three_mount(canvas_id: &str, theme: ResolvedTheme) {
    let Some(el) = window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(canvas_id)) else {
        return;
    };
    call_global_method("PermanentThree", "mount", &[el.into(), JsValue::from_str(theme.as_str())]);
}

pub fn three_set_problem(geometry: &ProblemGeometry, matching: Option<&[u32]>, valid: bool) {
    let Some(geo_js) = to_js(geometry) else { return };
    let match_js = matching.and_then(|m| to_js(&m)).unwrap_or(JsValue::NULL);
    call_global_method("PermanentThree", "setProblem", &[geo_js, match_js, JsValue::from_bool(valid)]);
}

pub fn three_set_matching(matching: Option<&[u32]>, valid: bool) {
    let match_js = matching.and_then(|m| to_js(&m)).unwrap_or(JsValue::NULL);
    call_global_method("PermanentThree", "setMatching", &[match_js, JsValue::from_bool(valid)]);
}

pub fn three_set_theme(theme: ResolvedTheme) {
    call_global_method("PermanentThree", "setTheme", &[JsValue::from_str(theme.as_str())]);
}

pub fn three_set_view_preset(preset: &str) {
    call_global_method("PermanentThree", "setViewPreset", &[JsValue::from_str(preset)]);
}

pub fn three_set_projection(mode: &str) {
    call_global_method("PermanentThree", "setProjection", &[JsValue::from_str(mode)]);
}

pub fn three_set_slice_z(slice_z: Option<usize>) {
    let val = match slice_z {
        Some(z) => JsValue::from_f64(z as f64),
        None => JsValue::NULL,
    };
    call_global_method("PermanentThree", "setSliceZ", &[val]);
}

pub fn three_set_show_periodic(show: bool) {
    call_global_method("PermanentThree", "setShowPeriodic", &[JsValue::from_bool(show)]);
}

pub fn three_dispose() {
    call_global_method("PermanentThree", "dispose", &[]);
}

pub fn copy_to_clipboard(text: &str) {
    if let Some(w) = window() {
        let clipboard = w.navigator().clipboard();
        let _ = clipboard.write_text(text);
    }
}

pub fn download_json(filename: &str, json_text: &str) {
    let Some(w) = window() else { return };
    let Some(doc) = w.document() else { return };
    let parts = Array::new();
    parts.push(&JsValue::from_str(json_text));
    let opts = BlobPropertyBag::new();
    opts.set_type("application/json");
    let Ok(blob) = Blob::new_with_str_sequence_and_options(&parts, &opts) else { return };
    let Ok(url) = Url::create_object_url_with_blob(&blob) else { return };
    let Ok(anchor) = doc.create_element("a") else { return };
    let Ok(anchor) = anchor.dyn_into::<HtmlAnchorElement>() else { return };
    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();
    let _ = Url::revoke_object_url(&url);
}
