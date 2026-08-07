//! pester: a tiny agent harness attached to one Discord channel.
//!
//! The crate splits along the seams: [`core`] holds the message vocabulary,
//! [`client`] the provider abstraction, [`responses`] the OpenAI
//! Responses-API client (platform or ChatGPT Codex backend), [`data`] the
//! data directory and system-prompt pipeline, [`tools`] the jailed file
//! tools and the clock, [`agent`] the turn loop, [`discord`] the endpoint,
//! and [`web`] the status UI.

pub mod agent;
pub mod client;
pub mod config;
pub mod core;
pub mod data;
pub mod daynote;
pub mod discord;
pub mod reminders;
pub mod responses;
pub mod sse;
pub mod status;
pub mod template;
pub mod timestamp;
pub mod tools;
pub mod web;
