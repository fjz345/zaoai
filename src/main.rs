#![allow(dead_code)]
#![allow(unreachable_code)]
#![allow(unused)]
#![allow(unused_mut)]
#![allow(clippy::redundant_closure)]
/*
Video -> Audio convert: ffmpeg
Digial processing: https://crates.io&/crates/&bliss-audio
Digial processing: https://crates.io/crates/fundsp&
Play audio: https://crates.io/crates/kira
*/

/*
Goal of this application:
Input a anime video file, add chapter timestamps for it for OP start/end & ED start/end.
This will be done by analyzing mainly the audio.

Features TODO list:
Support Audio Formats
    * mkv
    * mp4
*/

/*
    TODO:
    Create ML AI
        * Save/load weights
        * Dropout neurons
        * Cross-validation
        * MNIST
        * Softmax output
        * Eigen vectors? (reduce amount of input tensors)
        * Gradient dedcent momentum & decay
        * Add noise
        * Visualize output over time
    Add chapters to video file
    Gather training data
    Figure out how to train AI
*/

/*
ML Steps:
1. Data Preparation — Inspect and Prepare a Data Set
2. Define Model Validation Strategy — splitting data in train, validation and test set
3. Model development — building three different models using the sklearn library in Python: random forest, decision tree, logistic regression.
4. Model evaluation and fine-tuning (Hyperparameter Tuning) using GridSearch cross-validation
5. Model selection
6. Final Model evaluation
*/
mod filesystem;
mod graphviz;
mod graphviz_examples;
mod simpletest;
mod zneural_network;

use crate::filesystem::save_string_to_file;
use crate::graphviz::*;
use crate::graphviz_examples::*;
use crate::layer::*;
use crate::neuralnetwork::*;
use crate::simpletest::*;
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::*;

use soloud::*;
//use symphonia::core::sample;
use symphonia::core::units::TimeBase;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision};
use symphonia::core::probe::Hint;

static NN_GRAPH_LAYOUT_FILEPATH: &'static str = "zaoai_nn_layout.dot";

fn main() -> Result<(), SoloudError> {
    let nn_structure: GraphStructure = GraphStructure::new(&[2, 3, 2]);
    let mut nntest: NeuralNetwork = NeuralNetwork::new(nn_structure);
    nntest.validate();

    // parse_test();
    // print_test();
    // output_test();
    // graph_test();
    //output_exec_from_test();
    let graph_params: GenerateGraphParams = GenerateGraphParams { layer_spacing: 2.2 };
    let graph_layout = generate_nn_graph_layout_string(&nntest.graph_structure, &graph_params);
    // let graph_layout = generate_nn_graph_weight_bias_string(
    //     &nntest.graph_structure,
    //     &graph_params,
    //     &nntest.get_layers(),
    // );

    save_string_to_file(&graph_layout, NN_GRAPH_LAYOUT_FILEPATH);

    test_nn(&mut nntest);
    // TestNNOld(&mut nntest);

    return Ok(());

    let nn_structure: GraphStructure = GraphStructure::new(&[2, 3, 2]);
    let mut nn: NeuralNetwork = NeuralNetwork::new(nn_structure);
    nn.validate();

    nn.print();

    let mut datapoint = DataPoint {
        inputs: [2.0, 3.0],
        expected_outputs: [2.0, 3.0],
    };
    println!("Before: {:?}", datapoint.inputs);
    nn.calculate_outputs(&mut datapoint.inputs[..]);
    println!("After: {:?}", datapoint.inputs);

    return Ok(());

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

// returns a array with samples and the sample rate
fn decode_samples_from_file(path: &Path) -> (Vec<f32>, u32) {
    println!("###############################");
    println!(
        "[0/6] Start fetching samples for <{}>",
        &path.to_string_lossy()
    );

    // Open the media source.
    let src = File::open(&path).expect("failed to open media");

    println!("[1/6] Creating MediaSourceStream");
    // Create the media source stream.
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // Create a probe hint using the file's extension. [Optional]
    let mut hint = Hint::new();
    hint.with_extension("mp3");

    // Use the default options for metadata and format readers.
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    println!("[2/6] Creating ProbeResult");
    // Probe the media source.
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)
        .expect("unsupported format");

    // Get the instantiated format reader.
    let mut format = probed.format;

    println!("[3/6] Finding Track");
    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .expect("no supported audio tracks");

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    // Track data
    let mut sample_rate: u32 = 0;
    match track.codec_params.sample_rate {
        Some(v) => sample_rate = v,
        None => println!("sample_rate failed"),
    }
    let mut n_frames: u64 = 0;
    match track.codec_params.n_frames {
        Some(v) => n_frames = v,
        None => println!("n_frames failed"),
    }
    let mut time_base: TimeBase = TimeBase::default();
    match track.codec_params.time_base {
        Some(v) => time_base = v,
        None => println!("time_base failed"),
    }

    // Use the default options for the decoder.
    let dec_opts: DecoderOptions = Default::default();

    println!("[4/6] Creating Decoder for Track {}", track_id);
    // Create a decoder for the track.
    let decoder_result = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts);
    match &decoder_result {
        Ok(v) => (),
        Err(v) => panic!(
            "Codex Error: {} Codex was: {:?}",
            v, track.codec_params.codec
        ),
    }

    let mut decoder = decoder_result.unwrap();

    // The decode loop.
    let time_duration = time_base.calc_time(n_frames).seconds;
    println!("[5/6] Gathering Samples ({}s)", time_duration);

    // Determine 10 counter steps to show print
    let mut package_print_steps: Vec<u64> = Vec::new();
    let mut package_print_current_step: usize = 0;
    {
        let package_print_steps_num = 10;
        package_print_steps.reserve(package_print_steps_num + 1);
        for i in 0..package_print_steps_num {
            let n = (i as f32) / (package_print_steps_num as f32);
            let v = n as f32 * time_duration as f32;
            package_print_steps.push(v as u64);
        }
        package_print_steps.push(time_duration as u64);
    }

    let mut timestamp_counter: u64 = 0;
    let mut ret_samples: Vec<f32> = Vec::new();
    loop {
        // Get the next packet from the media format.
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::ResetRequired) => {
                // The track list has been changed. Re-examine it and create a new set of decoders,
                // then restart the decode loop. This is an advanced feature and it is not
                // unreasonable to consider this "the end." As of v0.5.0, the only usage of this is
                // for chained OGG physical streams.
                unimplemented!();
            }
            Err(err) => {
                // Handling this the real way somehow?
                if err.to_string() == String::from("end of stream") {
                    break;
                }
                // A unrecoverable error occured, halt decoding.
                panic!("{}", err);
            }
        };

        timestamp_counter += packet.dur;
        let cur_time = time_base.calc_time(timestamp_counter).seconds;
        if package_print_steps[package_print_current_step] == cur_time {
            // prevent it from being printed again
            package_print_steps[package_print_current_step] = 0;
            package_print_current_step =
                (package_print_current_step + 1) % package_print_steps.len();

            // Show % progress
            println!(
                "[{}%] Gathering Samples",
                100.0 * (cur_time as f32) / (time_duration as f32)
            );
        }

        // Consume any new metadata that has been read since the last packet.
        while !format.metadata().is_latest() {
            // Pop the old head of the metadata queue.
            let _metadata = format.metadata().pop();

            let md: MetadataRevision = _metadata.expect("MetadataRevision Failed");

            for tag in md.tags() {
                println!("Key: {}, Value: {}", tag.key, tag.value);
            }

            for vendordata in md.vendor_data() {
                println!("ident: {}, data: {}", vendordata.ident, vendordata.data[0]);
            }

            for visual in md.visuals() {
                for tag in visual.clone().tags {
                    println!("Key: {}, Value: {}", tag.key, tag.value);
                }
            }

            // Consume the new metadata at the head of the metadata queue.
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        match decoder.decode(&packet) {
            Ok(_decoded) => {
                match _decoded {
                    AudioBufferRef::F32(buf) => {
                        for &sample in buf.chan(0) {
                            ret_samples.push(sample);
                        }
                    }
                    _ => {
                        // Repeat for the different sample formats.
                        unimplemented!()
                    }
                }
            }
            Err(Error::IoError(_)) => {
                // The packet failed to decode due to an IO error, skip the packet.
                continue;
            }
            Err(Error::DecodeError(_)) => {
                // The packet failed to decode due to invalid data, skip the packet.
                continue;
            }
            Err(err) => {
                // An unrecoverable error occured, halt decoding.
                panic!("{}", err);
            }
        }
    }
    println!(
        "[6/6] Finished fetching samples for <{}> ({})",
        &path.to_string_lossy(),
        time_base.calc_time(timestamp_counter).seconds
    );

    (ret_samples, sample_rate)
}

static S_IS_DEBUG: i32 = 1;
fn init_soloud() -> Soloud {
    let sl = Soloud::default().expect("Soloud Init failed");

    if S_IS_DEBUG > 0 {
        sl.set_visualize_enable(true);
    }

    sl
}

use plotlib::page::Page;
use plotlib::repr::{Histogram, HistogramBins};
use plotlib::view::ContinuousView;

static S_HISTOGRAM_MAX_X: f64 = 0.5;
static S_HISTOGRAM_MAX_Y: f64 = 160.0;
fn sl_debug(sl: &Soloud) {
    let fft = sl.calc_fft();

    let mut vec64: Vec<f64> = Vec::new();
    vec64.reserve(fft.len());

    for f32_entry in fft {
        vec64.push(f32_entry as f64);
    }

    let histogram = Histogram::from_slice(&vec64, HistogramBins::Count(vec64.len()));
    let view = ContinuousView::new()
        .add(histogram)
        .x_range(0.0, S_HISTOGRAM_MAX_X)
        .y_range(0.0, S_HISTOGRAM_MAX_Y);

    if !Command::new("cmd")
        .arg("/C")
        .arg("cls")
        .status()
        .unwrap()
        .success()
    {
        panic!("cls failed....");
    }
    println!(
        "{}",
        Page::single(&view)
            .dimensions((60.0 * 4.7) as u32, 15)
            .to_text()
            .unwrap()
    );
}

use sonogram::*;
static S_SPECTOGRAM_PATH_DIR: &str = "";
static S_SPECTOGRAM_NUM_BINS: usize = 2048;
static S_SPECTOGRAM_WIDTH: usize = 512;
static S_SPECTOGRAM_HEIGHT: usize = 512;
fn save_spectrograph_as_png(
    path_dir: &String,
    path_filename: &String,
    data: &Vec<f32>,
    sample_rate: u32,
) {
    // Save the spectrogram to PNG.
    let mut path: PathBuf = PathBuf::new();
    path.push(&path_dir);
    path.push(&path_filename);
    let png_file = std::path::Path::new(&path);

    println!("###############################");
    println!(
        "[0/1] Start spectrograph to png <{}>",
        path.to_string_lossy()
    );

    // Build the model
    let mut spectrobuilder = SpecOptionsBuilder::new(S_SPECTOGRAM_NUM_BINS)
        .load_data_from_memory_f32(data.clone(), sample_rate)
        .build()
        .unwrap();

    // Compute the spectrogram giving the number of bins and the window overlap.
    let mut spectogram = spectrobuilder.compute();

    // Specify a colour gradient to use (note you can create custom ones)
    let mut gradient = ColourGradient::black_white_theme();

    spectogram
        .to_png(
            &png_file,
            FrequencyScale::Linear,
            &mut gradient,
            S_SPECTOGRAM_WIDTH,  // Width
            S_SPECTOGRAM_HEIGHT, // Height
        )
        .expect("Spectogram to png failed.");

    println!("[1/1] Finish spectrograph to png");
}
