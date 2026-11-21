//! Estimates the permanent of a nonnegative matrix to a requested accuracy.
//!
//! This crate is the algorithm: scaling, the temperature schedule, the plan
//! and the estimator. The population it works on lives in an
//! [`engine::Engine`], which other crates implement.

pub mod chain;
pub mod comm;
pub mod engine;
pub mod estimator;
pub mod exact;
pub mod problem;
pub mod sampler;
pub mod scaling;
pub mod stats;
pub mod tables;
