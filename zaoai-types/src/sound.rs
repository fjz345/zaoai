use std::fs::File;
use std::io::{BufReader, Read};
use std::mem;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::Context;
use anyhow::Result;
//use symphonia::core::sample;
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::{MetadataOptions, MetadataRevision};
use symphonia::core::probe::Hint;
use symphonia::core::units::TimeBase;

use symphonia::core::codecs::{
    CODEC_TYPE_AAC, CODEC_TYPE_FLAC, CODEC_TYPE_MP3, CODEC_TYPE_NULL, CODEC_TYPE_OPUS,
    CODEC_TYPE_PCM_F32LE, CODEC_TYPE_PCM_S16LE, CodecType,
};

const SUPPORTED_AUDIO_CODECS: &[CodecType] = &[
    CODEC_TYPE_MP3,
    CODEC_TYPE_AAC,
    CODEC_TYPE_FLAC,
    CODEC_TYPE_PCM_F32LE,
    CODEC_TYPE_PCM_S16LE,
    CODEC_TYPE_OPUS,
    // Add more as needed...
];

pub fn decode_audio_with_ffmpeg_f32(path: &str) -> Result<(Vec<f32>, u32)> {
    // Step 1: Extract sample rate using ffprobe
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let sample_rate_str = String::from_utf8_lossy(&output.stdout);
    let sample_rate: u32 = sample_rate_str
        .trim()
        .parse()
        .context("Failed to parse sample rate")?;

    // Step 2: Decode audio to raw float32 PCM (mono, sample_rate Hz)
    let mut child = Command::new("ffmpeg")
        .args([
            "-i",
            path,
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            &sample_rate.to_string(),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start ffmpeg")?;

    let mut raw_bytes = Vec::new();
    if let Some(out) = child.stdout.take() {
        let mut reader = BufReader::new(out);
        reader.read_to_end(&mut raw_bytes)?;
    }

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("ffmpeg decoding failed");
    }

    // Step 3: Convert bytes to f32 samples
    if raw_bytes.len() % 4 != 0 {
        anyhow::bail!("ffmpeg output size is not a multiple of 4 (f32 sample size)");
    }

    let mut samples = Vec::with_capacity(raw_bytes.len() / 4);
    for chunk in raw_bytes.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap(); // safe due to chunks_exact
        samples.push(f32::from_le_bytes(bytes));
    }

    Ok((samples, sample_rate))
}

pub fn decode_audio_with_ffmpeg_u8(path: &str) -> Result<(Vec<u8>, u32)> {
    // First, probe the file to get sample rate using ffprobe
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        anyhow::bail!(
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let sample_rate_str = String::from_utf8_lossy(&output.stdout);
    let sample_rate: u32 = sample_rate_str
        .trim()
        .parse()
        .context("Failed to parse sample rate")?;

    // Now decode the audio to mono PCM16LE WAV using that sample rate
    let mut child = Command::new("ffmpeg")
        .args([
            "-i",
            path,
            "-f",
            "wav",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            &sample_rate.to_string(),
            "-",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to start ffmpeg")?;

    let mut output_buf = Vec::new();
    if let Some(out) = child.stdout.take() {
        let mut reader = BufReader::new(out);
        reader.read_to_end(&mut output_buf)?;
    }

    let status = child.wait()?;
    if !status.success() {
        anyhow::bail!("ffmpeg decoding failed");
    }

    Ok((output_buf, sample_rate))
}

// returns a array with samples and the sample rate
pub fn decode_samples_audio_only_from_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    log::info!("###############################");
    log::info!(
        "[0/6] Start fetching samples for <{}>",
        &path.to_string_lossy()
    );

    let src = File::open(&path).with_context(|| format!("Failed to open: {}", path.display()))?;

    log::info!("[1/6] Creating MediaSourceStream");
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mkv");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    log::info!("[2/6] Creating ProbeResult");
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    for t in format.tracks() {
        let codec = t.codec_params.codec;
        eprintln!("Track index = {}, codec = {:?}", t.id, codec);
    }

    log::info!("[3/6] Finding Track");
    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| {
            let codec = t.codec_params.codec;
            codec != CODEC_TYPE_NULL && SUPPORTED_AUDIO_CODECS.contains(&codec)
        })
        .with_context(|| format!("No supported audio track found in file: {}", path.display()))?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut sample_rate: u32 = 0;
    match track.codec_params.sample_rate {
        Some(v) => sample_rate = v,
        None => log::info!("sample_rate failed"),
    }
    let mut n_frames: u64 = 0;
    match track.codec_params.n_frames {
        Some(v) => n_frames = v,
        None => log::info!("n_frames failed"),
    }
    // let mut time_base: TimeBase = TimeBase::default();
    // match track.codec_params.time_base {
    //     Some(v) => time_base = v,
    //     None => log::info!("time_base failed"),
    // }

    log::info!("[4/6] Creating Decoder for Track {}", track_id);
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    // The decode loop.
    log::info!("[5/6] Decoding...");
    let mut ret_samples = Vec::with_capacity(n_frames as usize);
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        if let Ok(decoded) = decoder.decode(&packet) {
            if let AudioBufferRef::F32(buf) = decoded {
                ret_samples.extend_from_slice(buf.chan(0));
            } else {
                anyhow::bail!("Non-f32 sample format not supported");
            }
        }
    }

    log::info!(
        "[6/6] Finished fetching samples for <{}>",
        &path.to_string_lossy(),
    );

    Ok((ret_samples, sample_rate))
}

// returns a array with samples and the sample rate
pub fn decode_samples_only_from_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    log::info!("###############################");
    log::info!(
        "[0/6] Start fetching samples for <{}>",
        &path.to_string_lossy()
    );

    let src = File::open(&path).with_context(|| format!("Failed to open: {}", path.display()))?;

    log::info!("[1/6] Creating MediaSourceStream");
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mkv");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    log::info!("[2/6] Creating ProbeResult");
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    log::info!("[3/6] Finding Track");
    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .with_context(|| "Could not find track")?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut sample_rate: u32 = 0;
    match track.codec_params.sample_rate {
        Some(v) => sample_rate = v,
        None => log::info!("sample_rate failed"),
    }
    let mut n_frames: u64 = 0;
    match track.codec_params.n_frames {
        Some(v) => n_frames = v,
        None => log::info!("n_frames failed"),
    }
    // let mut time_base: TimeBase = TimeBase::default();
    // match track.codec_params.time_base {
    //     Some(v) => time_base = v,
    //     None => log::info!("time_base failed"),
    // }

    log::info!("[4/6] Creating Decoder for Track {}", track_id);
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    // The decode loop.
    log::info!("[5/6] Decoding...");
    let mut ret_samples = Vec::with_capacity(n_frames as usize);
    while let Ok(packet) = format.next_packet() {
        if packet.track_id() != track_id {
            continue;
        }

        if let Ok(decoded) = decoder.decode(&packet) {
            if let AudioBufferRef::F32(buf) = decoded {
                ret_samples.extend_from_slice(buf.chan(0));
            } else {
                anyhow::bail!("Non-f32 sample format not supported");
            }
        }
    }

    log::info!(
        "[6/6] Finished fetching samples for <{}>",
        &path.to_string_lossy(),
    );

    Ok((ret_samples, sample_rate))
}

// returns a array with samples and the sample rate
pub fn decode_samples_from_file(path: &Path, read_metadata: bool) -> Result<(Vec<f32>, u32)> {
    log::info!("###############################");
    log::info!(
        "[0/6] Start fetching samples for <{}>",
        &path.to_string_lossy()
    );

    let src = File::open(&path)?;

    log::info!("[1/6] Creating MediaSourceStream");
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("mkv");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    log::info!("[2/6] Creating ProbeResult");
    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    // Get the instantiated format reader.
    let mut format = probed.format;

    log::info!("[3/6] Finding Track");
    // Find the first audio track with a known (decodeable) codec.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .with_context(|| "Could not find track")?;

    // Store the track identifier, it will be used to filter packets.
    let track_id = track.id;

    let mut sample_rate: u32 = 0;
    match track.codec_params.sample_rate {
        Some(v) => sample_rate = v,
        None => log::info!("sample_rate failed"),
    }
    let mut n_frames: u64 = 0;
    match track.codec_params.n_frames {
        Some(v) => n_frames = v,
        None => log::info!("n_frames failed"),
    }
    let mut time_base: TimeBase = TimeBase::default();
    match track.codec_params.time_base {
        Some(v) => time_base = v,
        None => log::info!("time_base failed"),
    }

    log::info!("[4/6] Creating Decoder for Track {}", track_id);
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    // The decode loop.
    let time_duration = time_base.calc_time(n_frames).seconds;
    log::info!("[5/6] Gathering Samples ({}s)", time_duration);

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
    let mut ret_samples = Vec::with_capacity(n_frames as usize);
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
            log::info!(
                "[{}%] Gathering Samples",
                100.0 * (cur_time as f32) / (time_duration as f32)
            );
        }

        // Consume any new metadata that has been read since the last packet.
        if read_metadata {
            while !format.metadata().is_latest() {
                // Pop the old head of the metadata queue.
                let _metadata = format.metadata().pop();

                let md: MetadataRevision = _metadata.with_context(|| "MetadataRevision Failed")?;

                for tag in md.tags() {
                    log::info!("Key: {}, Value: {}", tag.key, tag.value);
                }

                for vendordata in md.vendor_data() {
                    log::info!("ident: {}, data: {}", vendordata.ident, vendordata.data[0]);
                }

                for visual in md.visuals() {
                    for tag in visual.clone().tags {
                        log::info!("Key: {}, Value: {}", tag.key, tag.value);
                    }
                }

                // Consume the new metadata at the head of the metadata queue.
            }
        }

        // If the packet does not belong to the selected track, skip over it.
        if packet.track_id() != track_id {
            continue;
        }

        // Decode the packet into audio samples.
        let decoded = decoder.decode(&packet)?;
        match decoded {
            AudioBufferRef::F32(buf) => {
                for &sample in buf.chan(0) {
                    ret_samples.push(sample);
                }
            }
            _ => {
                // Repeat for the different sample formats.
                anyhow::bail!("unimplemented");
            }
        }
    }
    log::info!(
        "[6/6] Finished fetching samples for <{}> ({})",
        &path.to_string_lossy(),
        time_base.calc_time(timestamp_counter).seconds
    );

    Ok((ret_samples, sample_rate))
}

use sonogram::*;

use crate::spectrogram::S_SPECTROGRAM_NUM_BINS;
pub fn save_spectrograph_as_png(
    path: &PathBuf,
    data: &Vec<f32>,
    sample_rate: u32,
    out_dim: [usize; 2],
) {
    // Save the spectrogram to PNG.
    let png_file = std::path::Path::new(&path);

    log::info!("###############################");
    log::info!(
        "[0/1] Start spectrograph to png <{}>",
        path.to_string_lossy()
    );

    let mut spectrobuilder = SpecOptionsBuilder::new(S_SPECTROGRAM_NUM_BINS)
        .load_data_from_memory_f32(data.clone(), sample_rate)
        .build()
        .unwrap();
    let mut spectogram = spectrobuilder.compute();

    let mut gradient = ColourGradient::black_white_theme();
    spectogram
        .to_png(
            &png_file,
            FrequencyScale::Linear,
            &mut gradient,
            out_dim[0],
            out_dim[1],
        )
        .expect("Spectogram to png failed.");

    log::info!("[1/1] Finish spectrograph to png");
}

pub unsafe fn get_spectrogram_dims(spectrogram: &Spectrogram) -> (usize, usize) {
    unsafe {
        // Treat the pointer as a raw byte pointer
        let base = spectrogram as *const Spectrogram as *const u8;

        // Calculate offset of `width` and `height` fields
        let width_offset = mem::size_of::<Vec<f32>>();
        let height_offset = width_offset + mem::size_of::<usize>();

        // Read `width` and `height` from memory
        let width = *(base.add(width_offset) as *const usize);
        let height = *(base.add(height_offset) as *const usize);

        (width, height)
    }
}
