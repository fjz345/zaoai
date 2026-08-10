use anyhow::{Context, Result};
use std::{
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

use sonogram::{SpecOptionsBuilder, Spectrogram};

use crate::sound::decode_audio_with_ffmpeg_f32;

pub static S_SPECTROGRAM_SCALE: sonogram::FrequencyScale = sonogram::FrequencyScale::Log;
pub static S_SPECTROGRAM_NUM_BINS: usize = 2048;
pub static S_SPECTROGRAM_STEP_SIZE: usize = 2048;
pub const SPECTROGRAM_WIDTH: usize = 128;
pub const SPECTROGRAM_HEIGHT: usize = 32 as usize;

pub fn generate_spectrogram_ffmpeg(path: &Path, width: usize, height: usize) -> Result<Vec<f32>> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid path string"))?;

    // mode=combined: downmix to mono
    // color=intensity: output grayscale bytes (0-255)
    // scale=lin: linear frequency scale
    // legend=0: remove text/axes padding from the raw image
    // vflip: flips Y-axis so index 0 is 0Hz (matching sonogram array layout)
    let filter = format!(
        "aformat=channel_layouts=mono,showspectrumpic=s={}x{}:mode=combined:color=intensity:scale=lin:legend=0,vflip",
        width, height
    );

    let output = Command::new("ffmpeg")
        .args([
            "-v", "error", "-i", path_str, "-lavfi", &filter, "-f", "rawvideo", "-pix_fmt", "gray",
            "-",
        ])
        .output()
        .context("Failed to execute ffmpeg command")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    // Convert FFmpeg's 8-bit grayscale intensity into normalized f32 (0.0 - 1.0).
    // This perfectly mimics sonogram::to_buffer's output type and layout.
    let f32_buffer: Vec<f32> = output
        .stdout
        .into_iter()
        .map(|byte| byte as f32 / 255.0)
        .collect();

    Ok(f32_buffer)
}

pub fn generate_spectrogram(path: &PathBuf, num_spectrogram_bins: usize) -> Result<Spectrogram> {
    let (samples, sample_rate) = decode_audio_with_ffmpeg_f32(&path.to_str().unwrap())?;

    let mut spectrobuilder = SpecOptionsBuilder::new(num_spectrogram_bins)
        .load_data_from_memory_f32(samples, sample_rate)
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build spectrogram: {:?}", e))?;

    let spectrogram = spectrobuilder.compute();

    Ok(spectrogram)
}

pub fn save_spectrogram(
    buffer: Vec<f32>,
    width: usize,
    height: usize,
    path: impl AsRef<Path>,
) -> Result<()> {
    log::debug!("save_spectrogram [{},{}]", width, height);

    // Consuming Vec directly prevents cloning the giant buffer inside the tuple
    let data = (width, height, buffer);
    let bytes = bincode::encode_to_vec(data, BINCODE_CONFIG)?;

    let mut file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create file at {}", path.as_ref().display()))?;

    file.write_all(&bytes)?;

    Ok(())
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();
pub fn save_spectrogram_sonogram(
    spectrogram: &Spectrogram,
    width: usize,
    height: usize,
    path: impl AsRef<Path>,
) -> Result<()> {
    log::debug!("save_spectrogram [{},{}]", width, height);
    let spectrogram_buffer = spectrogram.to_buffer(sonogram::FrequencyScale::Linear, width, height);
    let data = (width, height, spectrogram_buffer);
    let bytes = bincode::encode_to_vec(data, BINCODE_CONFIG)?;

    if path.as_ref().exists() {
        log::info!("{}, already exists", path.as_ref().to_string_lossy());
    }
    let mut file = File::create(path.as_ref())
        .with_context(|| format!("Failed to create file at {}", path.as_ref().display()))?;
    file.write_all(&bytes)?;

    Ok(())
}

pub fn load_spectrogram(
    path: impl AsRef<Path>,
    out_width: &mut usize,
    out_height: &mut usize,
) -> Result<Spectrogram> {
    let bytes = std::fs::read(&path).with_context(|| format!("{}", path.as_ref().display()))?;
    let (width, height, buffer): (usize, usize, Vec<f32>) =
        bincode::decode_from_slice(&bytes, BINCODE_CONFIG).map(|(v, _)| v)?;

    #[allow(unused_mut)]
    let mut spectrogram = unsafe { create_spectrogram_unsafe(buffer, width, height) };

    let mut test_path = path.as_ref().to_path_buf();
    test_path.set_extension("png");
    spectrogram.to_png(
        &test_path,
        sonogram::FrequencyScale::Log,
        &mut sonogram::ColourGradient::black_white_theme(),
        width,
        height,
    )?;

    *out_width = width;
    *out_height = height;
    Ok(spectrogram)
}

pub unsafe fn create_spectrogram_unsafe(
    spec: Vec<f32>,
    width: usize,
    height: usize,
) -> Spectrogram {
    #[allow(dead_code)]
    struct SpectrogramRepr {
        spec: Vec<f32>,
        width: usize,
        height: usize,
    }

    let repr = SpectrogramRepr {
        spec,
        width,
        height,
    };

    unsafe { std::mem::transmute::<SpectrogramRepr, Spectrogram>(repr) }
}

pub unsafe fn resize_spectrogram(spectrogram: Spectrogram, w: usize, h: usize) -> Spectrogram {
    let resized_buffer = spectrogram.to_buffer(S_SPECTROGRAM_SCALE, w, h);
    unsafe { create_spectrogram_unsafe(resized_buffer, w, h) }
}
