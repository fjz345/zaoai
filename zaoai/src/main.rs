#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused)]
#![allow(unused_mut)]
#![allow(clippy::redundant_closure)]

// Video -> Audio convert: ffmpeg
// Digial processing: https://crates.io&/crates/&bliss-audio
// Digial processing: https://crates.io/crates/fundsp&
// Play audio: https://crates.io/crates/kira

// ML Steps:
// 1. Data Preparation — Inspect and Prepare a Data Set
// 2. Define Model Validation Strategy — splitting data in train, validation and test set
// 3. Model development — building three different models using the sklearn library in Python: random forest, decision tree, logistic regression.
// 4. Model evaluation and fine-tuning (Hyperparameter Tuning) using GridSearch cross-validation
// 5. Model selection
// 6. Final Model evaluation

mod app;
mod app_windows;
mod egui_ext;
mod error;
mod filesystem;
mod graphviz;
mod graphviz_examples;
mod mnist;
mod zneural_network;

use crate::app::*;
use crate::filesystem::save_string_to_file;
use crate::graphviz::*;
use crate::graphviz_examples::*;
use crate::layer::*;
use crate::neuralnetwork::*;
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::*;

use eframe::egui;
use std::env;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

static NN_GRAPH_LAYOUT_FILEPATH: &'static str = "zaoai_nn_layout.dot";

// Change the alias to `Box<dyn error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    env::set_var("RUST_BACKTRACE", "1");
    #[cfg(feature = "linux-profile")]
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(100)
        .blocklist(&["libc", "libgcc", "pthread", "vdso", "eframe"])
        .build()
        .unwrap();

    env::set_var("RUST_LOG", "debug"); // or "info" or "debug"
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // let path_testdir = String::from("test_files");
    // // let filename = String::from("mp3.mp3");
    // let filename = String::from("test0.mkv");
    // let mut path = std::path::PathBuf::from(path_testdir);
    // path.push(filename);
    // let (samples, sample_rate) = decode_samples_from_file(&path.as_path());

    // // let mut wav = audio::Wav::default();
    // // unsafe {
    // //     wav.load_raw_wav(&samples)?;
    // // }
    // // wav.set_volume(0.2);
    // // preview_sound_file(wav);

    // save_spectrograph_as_png(
    //     &PathBuf::from("").join("test2.png"),
    //     &samples,
    //     sample_rate,
    //     [512, 512],
    // );

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([2560.0, 1440.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ZaoAI",
        native_options,
        Box::new(move |cc: &eframe::CreationContext<'_>| {
            // This gives us image support:
            egui_extras::install_image_loaders(&cc.egui_ctx);

            #[cfg(feature = "serde")]
            {
                // Try to load saved state from storage
                if let Some(storage) = cc.storage {
                    if let Some(json) = storage.get_string(eframe::APP_KEY) {
                        match serde_json::from_str::<ZaoaiApp>(&json) {
                            Ok(app) => {
                                log::info!("Found previous app storage");
                                return Ok(Box::new(app));
                            }
                            Err(e) => {
                                log::error!("Could not parse ZaoaiApp json: {e}");
                            }
                        }
                    } else {
                        log::error!("Could not get storage string");
                    }
                } else {
                    log::error!("Could not find app storage");
                }
            }

            let app = ZaoaiApp::new(cc);
            Ok(Box::<ZaoaiApp>::new(app))
        }),
    );

    #[cfg(feature = "linux-profile")]
    {
        log::info!("Profiling exporting...");
        if let Ok(report) = guard.report().build() {
            let file = File::create("flamegraph.svg").unwrap();
            report.flamegraph(file).unwrap();
        };
    }

    Ok(())
}
