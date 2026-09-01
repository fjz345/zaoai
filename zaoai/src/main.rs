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
mod graphviz;
mod graphviz_examples;
mod mnist;
mod zneural_network;

use crate::app::*;
use crate::layer::*;
use crate::neuralnetwork_cpu::*;
use crate::zneural_network::*;

use eframe::egui;
use std::env;

// static NN_GRAPH_LAYOUT_FILEPATH: &'static str = "zaoai_nn_layout.dot";

// Change the alias to `Box<dyn error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    color_backtrace::install();

    env::set_var("RUST_BACKTRACE", "1");
    #[cfg(feature = "linux-profile")]
    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(100)
        .blocklist(&["libc", "libgcc", "pthread", "vdso", "eframe"])
        .build()
        .unwrap();

    env::set_var(
        "RUST_LOG",
        "debug,
        eframe::native::run=info, egui_winit=info, eframe::native=info",
    );
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([2560.0, 1440.0]),
        ..Default::default()
    };
    if let Err(e) = eframe::run_native(
        "ZaoAI",
        native_options,
        Box::new(move |cc: &eframe::CreationContext<'_>| {
            egui_extras::install_image_loaders(&cc.egui_ctx);

            // Persistent storage started bugging out, disabled for now.
            // #[cfg(feature = "serde")]
            // {
            //     if let Some(storage) = cc.storage {
            //         if let Some(json) = storage.get_string(eframe::APP_KEY) {
            //             let loaded_app = serde_json::from_str::<ZaoaiApp>(&json);
            //             match loaded_app {
            //                 Ok(app) => {
            //                     log::info!("Found previous app storage");
            //                     return Ok(Box::new(app));
            //                 }
            //                 Err(e) => {
            //                     log::error!("Failed to parse saved app state JSON: {e}. Ignoring and starting fresh.");
            //                     // no return here, fall through to create new app
            //                 }
            //             }
            //         } else {
            //             log::info!("No saved app state found. Starting fresh.");
            //         }
            //     } else {
            //         log::info!("No app storage available. Starting fresh.");
            //     }
            // }

            let app = ZaoaiApp::new(cc);
            Ok(Box::<ZaoaiApp>::new(app))
        }),
    ) {
        log::error!("Failed to start eframe: {}", e);
    };

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
