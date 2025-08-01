/*

{
  "file": "episode_03.mkv",
  "opening_start": {
    "frame": 2212,
    "timestamp": 92.312,
    "normalized": 0.17
  },
  "opening_end": {
    "frame": 3665,
    "timestamp": 152.750,
    "normalized": 0.28
  }
}

*/

pub mod ai_labels;
pub mod chapters;
pub mod file;
pub mod mkv;
pub mod sound;
pub mod spectrogram;
pub mod temp;
pub mod utils;

pub use sonogram::FrequencyScale;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::chapters::Chapters;

#[derive(Deserialize, Serialize, Debug)]
pub struct MkvMetadata {
    pub path: PathBuf,
    pub duration: f64,
    pub chapters: Chapters,
}
