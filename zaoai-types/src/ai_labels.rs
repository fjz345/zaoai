use anyhow::{Context, Result};
use rayon::prelude::*;
use sonogram::Spectrogram;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{fs, path};

use std::{
    fs::File,
    path::{Path, PathBuf},
    time::Duration,
};

use serde::{Deserialize, Serialize};

use crate::file::{EntryKind, list_dir, list_dir_all, relative_path_from_base};
use crate::mkv::process_mkv_file;
use crate::spectrogram::{S_SPECTROGRAM_NUM_BINS, generate_spectrogram_ffmpeg};
use crate::spectrogram::{generate_spectrogram, save_spectrogram};
use crate::{chapters::VideoMetadata, utils::ListDirSplit};

pub struct AnimeDataPoint {
    pub path: PathBuf,
    pub spectrogram: Spectrogram,
    pub expected_outputs: Vec<f32>,
}

pub const ZAOAI_LABEL_VERSION: u8 = 1;
#[derive(Deserialize, Serialize, Debug, Default, Clone)]
pub struct ZaoaiLabel {
    pub path: PathBuf,
    pub path_source: PathBuf,
    pub metadata: VideoMetadata,
    pub version: u8,

    //OP
    #[serde(with = "humantime_serde")]
    pub opening_start_time: Option<Duration>,
    #[serde(with = "humantime_serde")]
    pub opening_end_time: Option<Duration>,
    pub opening_start_frame: Option<u32>,
    pub opening_end_frame: Option<u32>,
    pub opening_start_normalized: Option<f64>,
    pub opening_end_normalized: Option<f64>,

    //ED
    #[serde(with = "humantime_serde")]
    pub ending_start_time: Option<Duration>,
    #[serde(with = "humantime_serde")]
    pub ending_end_time: Option<Duration>,
    pub ending_start_frame: Option<u32>,
    pub ending_end_frame: Option<u32>,
    pub ending_start_normalized: Option<f64>,
    pub ending_end_normalized: Option<f64>,
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
    limit: usize,
) -> Result<()> {
    let out_path_ref = out_path.as_ref();
    let path_source = &list_dir_split.path_source;

    log::info!(
        "ListDirSplit:\n\
     ├─ with chapters:    {}\n\
     ├─ without chapters: {}\n\
     └─ skipped:          {}",
        list_dir_split.num_with_chapters,
        list_dir_split.num_without_chapters,
        list_dir_split.num_skipped,
    );

    list_dir_split
        .with_chapters
        .par_iter()
        .take(limit)
        .for_each(|entry| {
            let path_buf = entry.as_ref();

            if !path_buf.is_file() {
                log::error!("Entry not a file, skipping");
                return;
            }

            let mkv_metadata = match process_mkv_file(entry) {
                Ok(m) => m,
                Err(e) => {
                    log::error!("process_mkv_file error on {}: {e}", path_buf.display());
                    return;
                }
            };

            let ((op_start, op_end), (ed_start, ed_end)) =
                mkv_metadata.extract_opening_and_ending_times();

            match (op_start.zip(op_end), ed_start.zip(ed_end)) {
                (Some((ops, ope)), Some((eds, ede))) => log::debug!(
                    "Extract OP&ED: {}\n\
        ├─ OP:    {}s-{}s\n\
        └─ ED:    {}s-{}s",
                    path_buf.as_os_str().display(),
                    ops.as_secs(),
                    ope.as_secs(),
                    eds.as_secs(),
                    ede.as_secs()
                ),
                (Some((ops, ope)), None) => log::debug!(
                    "Extract OP&ED: {}\n\
            ├─ OP:    {}s-{}s\n\
            └─ ED:    None",
                    path_buf.as_os_str().display(),
                    ops.as_secs(),
                    ope.as_secs()
                ),
                (None, Some((eds, ede))) => log::debug!(
                    "Extract OP&ED: {}\n\
        ├─ OP:    None\n\
        └─ ED:    {}s-{}s",
                    path_buf.as_os_str().display(),
                    eds.as_secs(),
                    ede.as_secs()
                ),
                (None, None) => {
                    log::debug!("No OP or ED found: {}", path_buf.as_os_str().display());
                    return;
                }
            }

            let video_metadata: VideoMetadata = mkv_metadata.into();
            let total_secs = video_metadata.duration.as_secs_f64();

            let label = ZaoaiLabel {
                path: path_buf.to_path_buf(),
                path_source: path_source.clone(),
                metadata: video_metadata,
                version: ZAOAI_LABEL_VERSION,
                opening_start_time: op_start,
                opening_end_time: op_end,
                opening_start_normalized: op_start.map(|f| f.as_secs_f64() / total_secs),
                opening_end_normalized: op_end.map(|f| f.as_secs_f64() / total_secs),
                ending_start_time: ed_start,
                ending_end_time: ed_end,
                ending_start_normalized: ed_start.map(|f| f.as_secs_f64() / total_secs),
                ending_end_normalized: ed_end.map(|f| f.as_secs_f64() / total_secs),
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
        let mut list_of_entries = list_dir_all(&path, true, None)?;

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

pub fn generate_zaoai_label_spectrograms_multithread(
    list: &[EntryKind],
    spectrogram_file_extension: &str,
    spectrogram_dim: [usize; 2],
) -> Result<()> {
    let mut files = Vec::new();
    collect_target_files(list, &mut files)?;
    log::info!("Files found for spectrogram generation ({}):", files.len());
    for (i, file) in files.iter().enumerate() {
        log::info!("{}", file.display());
        const MAX_DISPLAY: usize = 10;
        if i >= MAX_DISPLAY {
            log::info!("...");
            break;
        }
    }

    files.into_par_iter().for_each(|path| {
        if let Err(e) =
            process_single_zaoai_label(&path, spectrogram_file_extension, spectrogram_dim)
        {
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

pub fn process_single_zaoai_label(path: &Path, ext: &str, dim: [usize; 2]) -> Result<()> {
    let mut save_path = path.to_path_buf();
    let success = save_path.set_extension(ext);
    assert!(success);

    // Short-circuit early to avoid expensive FFT if file exists
    if save_path.exists() {
        log::info!("{}, already exists", save_path.display());
        return Ok(());
    }

    let zaoai_label = ZaoaiLabelsLoader::load_single(path)?;
    let spectro_buffer = generate_spectrogram_ffmpeg(&zaoai_label.path, dim[0], dim[1])?;

    save_spectrogram(spectro_buffer, dim[0], dim[1], &save_path)?;
    log::info!("Saved spectrogram: {}", save_path.display());

    Ok(())
}

//// Queue
///
/*
Find files
   │
   ├── spectrogram already exists → skip
   │
   └── spectrogram missing
          │
          ▼
    bounded path queue
          │
          ▼
    2 network workers
          │
          │ copy
          ▼
    bounded local-file queue
          │
          ▼
    8 FFmpeg workers
          │
          ▼
    save spectrogram
          │
          ▼
    temp file automatically deleted
*/
use crossbeam_channel::bounded;

#[derive(Clone, Copy)]
pub struct PipelineConfig {
    pub network_queue_size: usize,
    pub network_workers: usize,
    pub ffmpeg_queue_size: usize,
    pub ffmpeg_workers: usize,
    pub stall_timeout: Duration,
}

struct LocalAudioFile {
    original_path: PathBuf,
    temp_file: tempfile::NamedTempFile,
}

fn copy_worker(
    receiver: crossbeam_channel::Receiver<PathBuf>,
    sender: crossbeam_channel::Sender<LocalAudioFile>,
    temp_dir: Option<PathBuf>,
    active_counter: &AtomicUsize,
) {
    for zlbl_path in receiver {
        active_counter.fetch_add(1, Ordering::Relaxed);

        let zaoai_label = match ZaoaiLabelsLoader::load_single(&zlbl_path) {
            Ok(label) => label,
            Err(e) => {
                log::error!("Failed to load label {}:\n{:?}", zlbl_path.display(), e);
                active_counter.fetch_sub(1, Ordering::Relaxed);
                continue;
            }
        };

        match copy_to_local_temp(&zaoai_label.path, temp_dir.as_deref()) {
            Ok(temp_file) => {
                let item = LocalAudioFile {
                    original_path: zlbl_path,
                    temp_file,
                };
                active_counter.fetch_sub(1, Ordering::Relaxed);
                let _ = sender.send(item);
            }
            Err(e) => {
                log::error!(
                    "Failed to copy media for {} locally:\n{:?}",
                    zlbl_path.display(),
                    e
                );
                active_counter.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

fn copy_to_local_temp(path: &Path, temp_dir: Option<&Path>) -> Result<tempfile::NamedTempFile> {
    use std::fs::File;
    use std::io::copy;
    use tempfile::Builder;

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| format!(".{}", ext))
        .unwrap_or_default();

    let mut builder = Builder::new();
    builder.prefix("zaoai_audio_").suffix(&extension);

    let mut temp_file = match temp_dir {
        Some(dir) => builder.tempfile_in(dir),
        None => builder.tempfile(),
    }
    .context("Failed to create temporary file")?;

    let mut source =
        File::open(path).with_context(|| format!("Failed to open {}", path.display()))?;

    copy(&mut source, &mut temp_file)
        .with_context(|| format!("Failed to copy {}", path.display()))?;

    Ok(temp_file)
}

fn process_local_file(local_audio_path: &Path, save_path: &Path, dim: [usize; 2]) -> Result<()> {
    let spectro_buffer = generate_spectrogram_ffmpeg(local_audio_path, dim[0], dim[1])?;
    save_spectrogram(spectro_buffer, dim[0], dim[1], save_path)?;
    Ok(())
}

fn pipeline_status(
    file_q: usize,
    cur_net: usize,
    local_q: usize,
    cur_ff: usize,
    config: &PipelineConfig,
) -> String {
    format!(
        "Q [{}/{}] -> Network [{}/{}] -> Q [{}/{}] -> FFmpeg [{}/{}]",
        file_q,
        config.network_queue_size,
        cur_net,
        config.network_workers,
        local_q,
        config.ffmpeg_queue_size,
        cur_ff,
        config.ffmpeg_workers,
    )
}

pub fn generate_zaoai_label_spectrograms_queued_multithread(
    list: &[EntryKind],
    spectrogram_file_extension: &str,
    spectrogram_dim: [usize; 2],
    custom_temp_dir: Option<PathBuf>,
    config: PipelineConfig,
    limit: usize,
) -> Result<()> {
    log::debug!("QUEUE_SIZE_NETWORK: {}", config.network_queue_size);
    log::debug!("NETWORK_WORKERS: {}", config.network_workers);
    log::debug!("QUEUE_SIZE_FFMPEG: {}", config.ffmpeg_queue_size);
    log::debug!("FFMPEG_WORKERS: {}", config.ffmpeg_workers);

    let mut files = Vec::new();
    collect_target_files(list, &mut files)?;

    let mut pending_files: Vec<_> = files
        .into_iter()
        .filter(|path| !path.with_extension(spectrogram_file_extension).exists())
        .collect();

    if limit > 0 {
        pending_files.truncate(limit);
    }

    let total_files = pending_files.len();
    log::info!("Files found for spectrogram generation ({}):", total_files);

    let (file_tx, file_rx) = bounded::<PathBuf>(config.network_queue_size);
    let (local_tx, local_rx) = bounded::<LocalAudioFile>(config.ffmpeg_queue_size);

    let active_network = AtomicUsize::new(0);
    let active_ffmpeg = AtomicUsize::new(0);
    let processed_count = AtomicUsize::new(0);
    let is_done = AtomicBool::new(false);

    std::thread::scope(|scope| -> Result<()> {
        let mon_file_rx = file_rx.clone();
        let mon_local_rx = local_rx.clone();
        let mon_net = &active_network;
        let mon_ff = &active_ffmpeg;
        let mon_processed = &processed_count;
        let mon_done = &is_done;

        scope.spawn(move || {
            let mut last_processed = 0;
            let mut stall_timer = Duration::ZERO;
            let tick = Duration::from_secs(2);

            while !mon_done.load(Ordering::Relaxed) {
                let cur_net = mon_net.load(Ordering::Relaxed);
                let cur_ff = mon_ff.load(Ordering::Relaxed);
                let file_q = mon_file_rx.len();
                let local_q = mon_local_rx.len();
                let cur_processed = mon_processed.load(Ordering::Relaxed);

                log::info!(
                    "Pipeline | {}",
                    pipeline_status(file_q, cur_net, local_q, cur_ff, &config)
                );
                if (cur_net > 0 || cur_ff > 0) && cur_processed == last_processed {
                    stall_timer += tick;
                    if stall_timer >= config.stall_timeout {
                        log::warn!(
                            "Pipeline stalled for {:?} | {}",
                            config.stall_timeout,
                            pipeline_status(file_q, cur_net, local_q, cur_ff, &config),
                        );
                        stall_timer = Duration::ZERO;
                    }
                } else {
                    stall_timer = Duration::ZERO;
                    last_processed = cur_processed;
                }

                std::thread::sleep(tick);
            }
        });

        scope.spawn(move || {
            for path in pending_files {
                let _ = file_tx.send(path);
            }
            drop(file_tx);
        });

        for _ in 0..config.network_workers {
            let rx = file_rx.clone();
            let tx = local_tx.clone();
            let t_dir = custom_temp_dir.clone();
            let net_counter = &active_network;

            scope.spawn(move || {
                copy_worker(rx, tx, t_dir, net_counter);
            });
        }

        // Dropping these handles closes the channels when threads finish sending
        drop(local_tx);
        drop(file_rx);

        let ffmpeg_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.ffmpeg_workers)
            .build()?;

        let ff_counter = &active_ffmpeg;
        let p_counter = &processed_count;

        ffmpeg_pool.install(|| {
            local_rx.into_iter().par_bridge().for_each(|item| {
                ff_counter.fetch_add(1, Ordering::Relaxed);

                let save_path = item
                    .original_path
                    .with_extension(spectrogram_file_extension);

                if let Err(e) =
                    process_local_file(item.temp_file.path(), &save_path, spectrogram_dim)
                {
                    log::error!(
                        "Failed processing {}:\n{:?}",
                        item.original_path.display(),
                        e
                    );
                } else {
                    log::debug!("Saved spectrogram: {}", save_path.display());
                }

                ff_counter.fetch_sub(1, Ordering::Relaxed);
                p_counter.fetch_add(1, Ordering::Relaxed);
            });
        });

        is_done.store(true, Ordering::Relaxed);
        Ok(())
    })
}
