use anyhow::Result;
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use sonogram::{SpecCompute, SpecOptionsBuilder, Spectrogram};

use crate::sound::decode_samples_from_file;

pub fn generate_spectogram(path: &PathBuf, num_spectogram_bins: usize) -> Spectrogram {
    let (samples, sample_rate) = decode_samples_from_file(&path.as_path());

    let mut spectrobuilder = SpecOptionsBuilder::new(num_spectogram_bins)
        .load_data_from_memory_f32(samples, sample_rate)
        .build()
        .unwrap();
    let mut spectogram = spectrobuilder.compute();

    spectrobuilder.compute()
}

const BINCODE_CONFIG: bincode::config::Configuration = bincode::config::standard();
pub fn save_spectrogram(
    spectogram: &Spectrogram,
    width: usize,
    height: usize,
    path: impl AsRef<Path>,
) -> Result<()> {
    let spectrogram_buffer = spectogram.to_buffer(sonogram::FrequencyScale::Linear, width, height);
    let data = (spectrogram_buffer, width, height);
    let bytes = bincode::encode_to_vec(data, BINCODE_CONFIG)?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}

pub fn load_spectrogram(path: impl AsRef<Path>) -> Result<Spectrogram> {
    let bytes = std::fs::read(path)?;
    let (buffer, width, height): (Vec<f32>, usize, usize) =
        bincode::decode_from_slice(&bytes, BINCODE_CONFIG).map(|(v, _)| v)?;

    let spectrogram = create_spectrogram_unsafe(buffer, width, height);
    Ok(spectrogram)
}

pub fn create_spectrogram_unsafe(spec: Vec<f32>, width: usize, height: usize) -> Spectrogram {
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
