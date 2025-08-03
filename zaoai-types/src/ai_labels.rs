use anyhow::{Context, Result};
use sonogram::Spectrogram;
use std::io::{Read, Write};
use std::sync::Arc;
use std::{fs, thread};

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

pub fn collect_zaoai_labels(
    list_dir_split: &ListDirSplit,
    out_path: impl AsRef<Path>,
) -> Result<()> {
    return collect_zaoai_labels_multithread(list_dir_split, out_path);

    #[allow(unreachable_code)]
    let path_source = &list_dir_split.path_source.clone();
    for entry_with_chapters in &list_dir_split.with_chapters {
        let path_buf = entry_with_chapters.as_ref();
        let zaoai_label = if path_buf.is_file() {
            if path_buf.is_file() {
                let b = process_mkv_file(&entry_with_chapters);
                match b {
                    Ok(mkv_metadata) => {
                        let (Some(op_start), Some(op_end)) = mkv_metadata.extract_opening_times()
                        else {
                            // Same effect as if zaoai_label was None
                            return Ok(());
                        };

                        let opening_start_time = op_start;
                        let opening_end_time = op_end;

                        let video_metadata: VideoMetadata = mkv_metadata.into();
                        let total_secs = video_metadata.duration.as_secs_f64();

                        let ai_label = ZaoaiLabel {
                            path: path_buf.clone(),
                            path_source: path_source.clone(),
                            metadata: video_metadata,
                            version: ZAOAI_LABEL_VERSION,
                            opening_start_time: Some(opening_start_time),
                            opening_end_time: Some(opening_end_time),
                            opening_start_frame: None,
                            opening_end_frame: None,
                            // Not sure if it should be div_eclid or div_ceil
                            opening_start_normalized: Some(
                                opening_start_time.as_secs_f64() / total_secs,
                            ),
                            opening_end_normalized: Some(
                                opening_end_time.as_secs_f64() / total_secs,
                            ),
                            ..Default::default()
                        };
                        Some(ai_label)
                    }
                    Err(e) => {
                        println!("{e}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(label) = zaoai_label {
            // println!("path_soruce: {}", label.path_source.display());
            // println!("path_buf: {}", path_buf.display());

            let relative_path = relative_path_from_base(path_buf, &label.path_source)?;
            // println!("relative: {}", Path::new(relative_path).display());

            let output_path = out_path.as_ref().join(relative_path).with_extension("zlbl");

            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            println!("{}", output_path.display());
            if output_path.exists() {
                eprintln!(
                    "Warning: Output file already exists and will be overwritten: {}",
                    output_path.display()
                );
            }

            let mut file = File::create(&output_path)?;
            let json = serde_json::to_string_pretty(&label)?;

            writeln!(file, "{}", json)?;
            println!("Wrote: {}", output_path.display());
        }
    }

    Ok(())
}

pub fn collect_zaoai_labels_multithread(
    list_dir_split: &ListDirSplit,
    out_path: impl AsRef<Path>,
) -> Result<()> {
    let out_path = out_path.as_ref().to_path_buf(); // clone for thread move
    let path_source = list_dir_split.path_source.clone();

    std::thread::scope(|scope| {
        let mut handles = vec![];

        for entry in list_dir_split.with_chapters.iter().cloned() {
            let path_source = path_source.clone();
            let out_path = out_path.clone();

            let handle = scope.spawn(move || -> Result<(), anyhow::Error> {
                let path_buf = entry.as_ref();

                if !path_buf.is_file() {
                    return Ok(()); // skip non-files
                }

                let mkv_metadata = match process_mkv_file(&entry) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("process_mkv_file error: {e}");
                        return Ok(());
                    }
                };

                let (Some(op_start), Some(op_end)) = mkv_metadata.extract_opening_times() else {
                    // Same effect as if zaoai_label was None
                    return Ok(());
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
                    opening_start_frame: None,
                    opening_end_frame: None,
                    opening_start_normalized: Some(op_start.as_secs_f64() / total_secs),
                    opening_end_normalized: Some(op_end.as_secs_f64() / total_secs),
                    ..Default::default()
                };

                let relative_path = relative_path_from_base(path_buf, &label.path_source)
                    .with_context(|| format!("Failed to compute relative path"))?;
                let output_path = out_path.join(relative_path).with_extension("zlbl");

                if let Some(parent) = output_path.parent() {
                    fs::create_dir_all(parent)?;
                }

                if output_path.exists() {
                    eprintln!(
                        "Warning: Output file already exists: {}",
                        output_path.display()
                    );
                }

                let mut file = File::create(&output_path)?;
                let json = serde_json::to_string_pretty(&label)?;
                writeln!(file, "{}", json)?;

                println!("Wrote: {}", output_path.display());
                Ok(())
            });

            handles.push(handle);
        }

        for handle in handles {
            if let Err(e) = handle.join().expect("thread panicked") {
                eprintln!("Worker failed: {e}");
            }
        }

        Ok::<(), anyhow::Error>(())
    })?;

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
                println!("EntryKind::Other not supported")
            }
        }
    }

    Ok(())
}

pub fn generate_zaoai_label_spectrograms_multithread(
    list: &[EntryKind],
    spectrogram_file_extension: &String,
    spectrogram_dim: [usize; 2],
) -> Result<()> {
    let extension_arc = Arc::new(spectrogram_file_extension.clone());

    let mut count = 0;

    thread::scope(|scope| {
        let mut handles = vec![];

        for entry in list {
            count += 1;

            if count >= 10 {
                break;
            }
            let spectrogram_file_extension = Arc::clone(&extension_arc);
            match entry {
                EntryKind::File(path_buf) => {
                    let path_buf = path_buf.clone();
                    let dim = spectrogram_dim;

                    let handle = scope.spawn(move || {
                        if path_buf.extension().unwrap_or_default() == "zlbl" && path_buf.is_file()
                        {
                            match ZaoaiLabelsLoader::load_single(&path_buf) {
                                Ok(zaoai_label) => {
                                    match generate_spectrogram(
                                        &zaoai_label.path,
                                        S_SPECTROGRAM_NUM_BINS,
                                    ) {
                                        Ok(specto) => {
                                            let mut save_path = path_buf.clone();
                                            let success = save_path
                                                .set_extension(&*spectrogram_file_extension);
                                            assert!(success);
                                            save_spectrogram(&specto, dim[0], dim[1], &save_path)?;
                                            log::info!(
                                                "Saved spectrogram: {}",
                                                save_path.display()
                                            );
                                        }
                                        Err(e) => {
                                            log::error!(
                                                "Spectrogram error on file:\n{}\nError: {:?}",
                                                zaoai_label.path.display(),
                                                e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!(
                                        "Label load error on file:\n{}\nError: {:?}",
                                        path_buf.display(),
                                        e
                                    );
                                }
                            }
                        }
                        Ok::<(), anyhow::Error>(())
                    });
                    handles.push(handle);
                }

                EntryKind::Directory(path_buf) => {
                    let path_buf = path_buf.clone();
                    let dim = spectrogram_dim;

                    let handle = scope.spawn(move || {
                        let dir_list = list_dir(&path_buf, true)?;
                        generate_zaoai_label_spectrograms_multithread(
                            &dir_list,
                            &spectrogram_file_extension,
                            dim,
                        )
                    });
                    handles.push(handle);
                }

                EntryKind::Other(_) => {
                    eprintln!("EntryKind::Other not supported");
                }
            }
        }

        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    })
}

pub struct AnimeDataPoint {
    pub path: PathBuf,
    pub spectrogram: Spectrogram,
    pub expected_outputs: Vec<f32>,
}
