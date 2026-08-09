use anyhow::{Context, Result};
use rayon::prelude::*;
use sonogram::Spectrogram;
use std::fs;
use std::io::{Read, Write};

use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::file::{EntryKind, list_dir, list_dir_all, relative_path_from_base};
use crate::mkv::process_mkv_file;
use crate::spectrogram::S_SPECTROGRAM_NUM_BINS;
use crate::spectrogram::{generate_spectrogram, save_spectrogram};
use crate::{chapters::VideoMetadata, utils::ListDirSplit};

pub const ZAOAI_LABEL_VERSION: u8 = 1;
#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub struct ZaoaiLabel {
    pub path: PathBuf,
    pub path_source: PathBuf,
    pub metadata: VideoMetadata,
    pub version: u8,

    #[serde(with = "humantime_serde")]
    pub opening_start_time: Option<Duration>,
    #[serde(with = "humantime_serde")]
    pub opening_end_time: Option<Duration>,
    pub opening_start_frame: Option<u32>,
    pub opening_end_frame: Option<u32>,
    pub opening_start_normalized: Option<f64>,
    pub opening_end_normalized: Option<f64>,
}

impl ZaoaiLabel {
    pub fn has_opening(&self) -> bool {
        self.opening_start_frame.is_some() && self.opening_end_frame.is_some()
    }

    pub fn expected_outputs(&self) -> Vec<f32> {
        let mut start_normalized = None;
        let mut end_normalized = None;
        if let Some(t0) = self.opening_start_normalized {
            start_normalized = Some(t0);
        }
        if let Some(t1) = self.opening_end_normalized {
            end_normalized = Some(t1);
        }

        let start = start_normalized.expect("failed to get start normalized");
        let end = end_normalized.expect("failed to get end normalized");

        vec![start as f32, end as f32]
    }
}

pub fn collect_zaoai_labels_multithread(
    list_dir_split: &ListDirSplit,
    out_path: impl AsRef<Path> + Sync,
) -> Result<()> {
    let out_path_ref = out_path.as_ref();
    let path_source = &list_dir_split.path_source;

    list_dir_split.with_chapters.par_iter().for_each(|entry| {
        let path_buf = entry.as_ref();

        if !path_buf.is_file() {
            return;
        }

        let mkv_metadata = match process_mkv_file(entry) {
            Ok(m) => m,
            Err(e) => {
                log::error!("process_mkv_file error on {}: {e}", path_buf.display());
                return;
            }
        };

        let (Some(op_start), Some(op_end)) = mkv_metadata.extract_opening_times() else {
            return;
        };

        let video_metadata: VideoMetadata = mkv_metadata.into();
        let total_secs = video_metadata.duration.as_secs_f64();

        let label = ZaoaiLabel {
            path: path_buf.to_path_buf(),
            path_source: path_source.clone(),
            metadata: video_metadata,
            version: ZAOAI_LABEL_VERSION,
            opening_start_time: Some(op_start),
            opening_end_time: Some(op_end),
            opening_start_normalized: Some(op_start.as_secs_f64() / total_secs),
            opening_end_normalized: Some(op_end.as_secs_f64() / total_secs),
            ..Default::default()
        };

        let relative_path = match relative_path_from_base(path_buf, path_source) {
            Ok(p) => p,
            Err(e) => {
                log::error!(
                    "Failed to compute relative path for {}: {e}",
                    path_buf.display()
                );
                return;
            }
        };

        let output_path = out_path_ref.join(relative_path).with_extension("zlbl");

        if let Some(parent) = output_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                log::error!("Failed to create directory {}: {e}", parent.display());
                return;
            }
        }

        if output_path.exists() {
            log::warn!(
                "Output file already exists and will be overwritten: {}",
                output_path.display()
            );
        }

        match File::create(&output_path) {
            Ok(mut file) => match serde_json::to_string_pretty(&label) {
                Ok(json) => {
                    if let Err(e) = writeln!(file, "{}", json) {
                        log::error!("Failed to write to {}: {e}", output_path.display());
                    } else {
                        log::info!("Wrote: {}", output_path.display());
                    }
                }
                Err(e) => {
                    log::error!(
                        "Failed to serialize JSON for {}: {e}",
                        output_path.display()
                    );
                }
            },
            Err(e) => {
                log::error!("Failed to create file {}: {e}", output_path.display());
            }
        }
    });

    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ZaoaiLabelsLoader {
    pub path_source: PathBuf,
    pub len: usize,
    pub label_file_paths: Vec<PathBuf>,
    pub label_input_dim: Option<[usize; 2]>,
}

impl ZaoaiLabelsLoader {
    pub fn load_single(path: impl AsRef<Path>) -> Result<ZaoaiLabel> {
        assert_eq!(path.as_ref().is_file(), true);
        assert_eq!(path.as_ref().extension().unwrap(), "zlbl");

        let label = Self::load_zaoai_label(path)?;
        Ok(label)
    }

    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let mut list_of_entries = list_dir_all(&path, true)?;

        // filter zlbl
        list_of_entries = list_of_entries
            .iter()
            .filter(|a| a.is_file() && a.extension().unwrap() == "zlbl")
            .cloned()
            .collect();

        Ok(Self {
            path_source: path.as_ref().to_path_buf(),
            len: list_of_entries.len(),
            label_file_paths: list_of_entries,
            label_input_dim: None,
        })
    }

    fn load_zaoai_label(file_path: impl AsRef<Path>) -> Result<ZaoaiLabel> {
        let mut file = std::fs::File::open(file_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let zaoai_label: ZaoaiLabel = serde_json::from_str(&contents)?;

        Ok(zaoai_label)
    }

    pub fn load_zaoai_labels(&self) -> Result<Vec<ZaoaiLabel>> {
        let mut vec = Vec::new();
        for file_path in &self.label_file_paths {
            let label = Self::load_zaoai_label(file_path)
                .with_context(|| format!("Failed to load_zaoai_label {}", file_path.display()))?;
            vec.push(label);
        }

        Ok(vec)
    }
}

pub fn generate_zaoai_label_spectrograms(
    list: &Vec<EntryKind>,
    spectrogram_file_extension: &String,
    spectrogram_dim: [usize; 2],
) -> Result<()> {
    return generate_zaoai_label_spectrograms_multithread(
        list,
        spectrogram_file_extension,
        spectrogram_dim,
    );

    #[allow(unreachable_code)]
    for entry in list {
        match entry {
            EntryKind::File(path_buf) => {
                if path_buf.extension().unwrap() == "zlbl" {
                    assert_eq!(path_buf.is_file(), true);

                    // Load zaoai_label
                    let zaoai_label = ZaoaiLabelsLoader::load_single(path_buf)?;

                    let spectrogram =
                        generate_spectrogram(&zaoai_label.path, S_SPECTROGRAM_NUM_BINS);
                    match spectrogram {
                        Ok(specto) => {
                            let mut spectrogram_save_path = path_buf.clone();
                            let success =
                                spectrogram_save_path.set_extension(spectrogram_file_extension);
                            assert!(success);

                            save_spectrogram(
                                &specto,
                                spectrogram_dim[0],
                                spectrogram_dim[1],
                                &spectrogram_save_path,
                            )?;

                            log::info!("Saved spectrogram: {}", spectrogram_save_path.display());
                        }
                        Err(e) => {
                            log::error!(
                                "Failed to generate spectrogram on file:\n{}\nError: {:?}",
                                zaoai_label.path.display(),
                                e
                            );
                        }
                    }
                }
            }
            EntryKind::Directory(path_buf) => {
                let dir_list_dir = list_dir(path_buf, true)?;
                generate_zaoai_label_spectrograms(
                    &dir_list_dir,
                    &spectrogram_file_extension,
                    spectrogram_dim,
                )?;
            }
            EntryKind::Other(_path_buf) => {
                log::info!("EntryKind::Other not supported")
            }
        }
    }

    Ok(())
}

pub fn generate_zaoai_label_spectrograms_multithread(
    list: &[EntryKind],
    spectrogram_file_extension: &str,
    spectrogram_dim: [usize; 2],
) -> Result<()> {
    let mut files = Vec::new();
    collect_target_files(list, &mut files)?;

    files.into_par_iter().for_each(|path| {
        if let Err(e) = process_single_file(&path, spectrogram_file_extension, spectrogram_dim) {
            log::error!("Failed processing {}:\n{:?}", path.display(), e);
        }
    });

    Ok(())
}

fn collect_target_files(list: &[EntryKind], files: &mut Vec<PathBuf>) -> Result<()> {
    for entry in list {
        match entry {
            EntryKind::File(path) => {
                if path.extension().unwrap_or_default() == "zlbl" && path.is_file() {
                    files.push(path.clone());
                }
            }
            EntryKind::Directory(path) => {
                let dir_list = list_dir(path, true)?;
                collect_target_files(&dir_list, files)?;
            }
            EntryKind::Other(other) => {
                log::error!("EntryKind::Other not supported: {}", other.display());
            }
        }
    }
    Ok(())
}

fn process_single_file(path: &Path, ext: &str, dim: [usize; 2]) -> Result<()> {
    let zaoai_label = ZaoaiLabelsLoader::load_single(path)?;

    let specto = generate_spectrogram(&zaoai_label.path, S_SPECTROGRAM_NUM_BINS)?;

    let mut save_path = path.to_path_buf();
    let success = save_path.set_extension(ext);
    assert!(success);

    save_spectrogram(&specto, dim[0], dim[1], &save_path)?;
    log::info!("Saved spectrogram: {}", save_path.display());

    Ok(())
}

pub struct AnimeDataPoint {
    pub path: PathBuf,
    pub spectrogram: Spectrogram,
    pub expected_outputs: Vec<f32>,
}
