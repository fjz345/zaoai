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
mod filesystem;
mod graphviz;
mod graphviz_examples;
mod mnist;
mod simpletest;
mod sound;
mod zneural_network;

use crate::app::*;
use crate::filesystem::save_string_to_file;
use crate::graphviz::*;
use crate::graphviz_examples::*;
use crate::layer::*;
use crate::neuralnetwork::*;
use crate::simpletest::*;
use crate::sound::decode_samples_from_file;
use crate::sound::init_soloud;
use crate::sound::save_spectrograph_as_png;
use crate::sound::sl_debug;
use crate::sound::S_IS_DEBUG;
use crate::sound::S_SPECTOGRAM_PATH_DIR;
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::*;

use eframe::egui;
use soloud::*;
use std::env;
use std::error;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

static NN_GRAPH_LAYOUT_FILEPATH: &'static str = "zaoai_nn_layout.dot";

// Change the alias to `Box<dyn error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

fn main() -> Result<()> {
    env::set_var("RUST_LOG", "debug"); // or "info" or "debug"
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let nn_structure: GraphStructure = GraphStructure::new(&[2, 3, 2], true);
    let mut nntest: NeuralNetwork = NeuralNetwork::new(nn_structure);
    nntest.validate();

    let graph_params: GenerateGraphParams = GenerateGraphParams { layer_spacing: 2.2 };
    let graph_layout = generate_nn_graph_layout_string(&nntest.graph_structure, &graph_params);

    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "ZaoAI",
        options,
        Box::new(|_cc| Ok(Box::<ZaoaiApp>::default())),
    );

    return Ok(());

    // Code create spectogram
    let path_testdir = String::from("test_files");
    // let filename = String::from("mp3.mp3");
    let filename = String::from("test0.mkv");

    let mut path = std::path::PathBuf::from(path_testdir);
    path.push(filename);

    let (samples, sample_rate) = decode_samples_from_file(&path.as_path());

    let sl: Soloud = init_soloud();

    let mut wav = audio::Wav::default();

    unsafe {
        wav.load_raw_wav(&samples)?;
    }

    wav.set_volume(0.2);

    save_spectrograph_as_png(
        &String::from(S_SPECTOGRAM_PATH_DIR),
        &String::from("test.png"),
        &samples,
        sample_rate,
    );

    sl.play(&wav); // calls to play are non-blocking, so we put the thread to sleep
    while sl.voice_count() > 0 {
        if S_IS_DEBUG > 0 {
            sl_debug(&sl);
        } else {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }

    Ok(())
}
