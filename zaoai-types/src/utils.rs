use crate::{
    chapters::extract_chapters,
    file::{EntryKind, list_dir},
};
use anyhow::{Context, Result};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self},
    io::Read,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

pub(crate) fn get_third_party_binary(name: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR will be the zaohelper/ path even when used from zaoai
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    base.join("third_party\\bin").join(name)
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct ListDirSplit {
    pub path_source: PathBuf,
    pub num_with_chapters: u32,
    pub num_without_chapters: u32,
    pub num_skipped: u32,
    pub with_chapters: Vec<EntryKind>,
    pub without_chapters: Vec<EntryKind>,
    pub skipped: Vec<EntryKind>,
}

impl ListDirSplit {
    pub fn from_file_json(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = fs::File::open(path)?;
        let mut json_str = String::new();
        file.read_to_string(&mut json_str)?;
        let new = serde_json::from_str::<Self>(&json_str)?;
        Ok(new)
    }
}
fn list_dir_split_status(
    completed: usize,
    total: usize,
    active: &[PathBuf],
    active_workers: usize,
    workers: usize,
) -> String {
    let percent = if total > 0 {
        completed as f64 / total as f64 * 100.0
    } else {
        100.0
    };

    let active_names = if active.is_empty() {
        "-".to_string()
    } else {
        active
            .iter()
            .map(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "-".to_string())
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };

    format!(
        "ListDirSplit | Workers [{}/{}] | Progress [{}/{}] {:.1}%\n\
         └─ Active: {}",
        active_workers, workers, completed, total, percent, active_names,
    )
}

pub fn collect_flat_files(
    list: &[EntryKind],
    cull_empty_folders: bool,
    flat_files: &mut Vec<EntryKind>,
    limit: usize,
) -> Result<()> {
    for item in list {
        if limit != 0 {
            if flat_files.len() >= limit {
                return Ok(());
            }
        }

        match item {
            EntryKind::File(_) => flat_files.push(item.clone()),
            EntryKind::Directory(path_buf) => {
                let entries = list_dir(path_buf, cull_empty_folders)
                    .with_context(|| format!("Failed to read directory: {}", path_buf.display()))?;
                collect_flat_files(&entries, cull_empty_folders, flat_files, limit)?;
            }
            EntryKind::Other(_) => {}
        }
    }
    Ok(())
}
pub fn list_dir_with_kind_has_chapters_split(
    list: &[EntryKind],
    cull_empty_folders: bool,
    limit: usize,
) -> Result<ListDirSplit> {
    let mut flat_files = Vec::new();
    collect_flat_files(list, cull_empty_folders, &mut flat_files, limit)?;

    let total = flat_files.len();
    let workers = rayon::current_num_threads();

    let completed = Arc::new(AtomicUsize::new(0));
    let active_workers = Arc::new(AtomicUsize::new(0));
    let stop_progress = Arc::new(AtomicBool::new(false));

    let pipeline_files = Arc::new(Mutex::new(Vec::<PathBuf>::new()));

    // ============================================================
    // PROGRESS MONITOR
    // ============================================================

    let progress_completed = Arc::clone(&completed);
    let progress_active_workers = Arc::clone(&active_workers);
    let progress_stop = Arc::clone(&stop_progress);
    let progress_pipeline_files = Arc::clone(&pipeline_files);

    let progress_thread = thread::spawn(move || {
        while !progress_stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(2));

            let done = progress_completed.load(Ordering::Relaxed);
            let active_count = progress_active_workers.load(Ordering::Relaxed);

            let active = {
                let files = progress_pipeline_files.lock().unwrap();
                files
                    .iter()
                    .map(|f| PathBuf::from(preview_name(Some(f))))
                    .collect::<Vec<PathBuf>>()
            };

            log::info!(
                "{}",
                list_dir_split_status(done, total, &active, active_count, workers,)
            );
        }
    });

    // ============================================================
    // RAYON WORKERS
    // ============================================================

    let split = flat_files
        .into_par_iter()
        .fold(ListDirSplit::default, |mut acc, item| {
            let path_buf = match &item {
                EntryKind::File(path) => path.clone(),

                _ => {
                    completed.fetch_add(1, Ordering::Relaxed);
                    acc.skipped.push(item);
                    return acc;
                }
            };

            // ----------------------------------------------------
            // Worker starts processing this file
            // ----------------------------------------------------

            active_workers.fetch_add(1, Ordering::Relaxed);

            {
                let mut active = pipeline_files.lock().unwrap();
                active.push(path_buf.clone());
            }

            // ----------------------------------------------------
            // Process the file
            // ----------------------------------------------------

            if path_buf
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("mkv"))
            {
                match extract_chapters(&path_buf) {
                    Ok(chapters) => {
                        if chapters.iter().next().is_some() {
                            acc.with_chapters.push(item);
                        } else {
                            acc.without_chapters.push(item);
                        }
                    }

                    Err(e) => {
                        log::error!("Chapter extract failed for {}: {e}", path_buf.display());

                        acc.skipped.push(item);
                    }
                }
            } else {
                acc.skipped.push(item);
            }

            // ----------------------------------------------------
            // Worker finished processing this file
            // ----------------------------------------------------

            {
                let mut active = pipeline_files.lock().unwrap();
                active.retain(|p| p != &path_buf);
            }

            active_workers.fetch_sub(1, Ordering::Relaxed);
            completed.fetch_add(1, Ordering::Relaxed);

            acc
        })
        .reduce(ListDirSplit::default, |mut a, mut b| {
            a.with_chapters.append(&mut b.with_chapters);
            a.without_chapters.append(&mut b.without_chapters);
            a.skipped.append(&mut b.skipped);
            a
        });

    // ============================================================
    // STOP PROGRESS MONITOR
    // ============================================================

    stop_progress.store(true, Ordering::Relaxed);
    progress_thread.join().ok();

    log::info!(
        "ListDirSplit complete: {}/{} files processed",
        completed.load(Ordering::Relaxed),
        total
    );

    Ok(split)
}

pub fn preview_name(path: Option<&Path>) -> String {
    const MAX_LEN: usize = 24;

    let name = match path.and_then(|p| p.file_name()) {
        Some(name) => name.to_string_lossy(),
        None => return "-".to_string(),
    };

    let char_count = name.chars().count();

    if char_count <= MAX_LEN {
        name.into_owned()
    } else {
        format!("{}…", name.chars().take(MAX_LEN - 1).collect::<String>())
    }
}
