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

pub fn collect_flat_files(
    list: &[EntryKind],
    cull_empty_folders: bool,
    flat_files: &mut Vec<EntryKind>,
    limit: Option<usize>,
) -> Result<()> {
    for item in list {
        if let Some(max) = limit {
            if flat_files.len() >= max {
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
    limit: Option<usize>,
) -> Result<ListDirSplit> {
    let mut flat_files = Vec::new();
    collect_flat_files(list, cull_empty_folders, &mut flat_files, limit)?;

    let split = flat_files
        .into_par_iter()
        .fold(ListDirSplit::default, |mut acc, item| {
            if let EntryKind::File(path_buf) = &item {
                if path_buf.extension().is_some_and(|ext| ext == "mkv") {
                    if let Some(mkv_file_str) = path_buf.to_str() {
                        match extract_chapters(mkv_file_str) {
                            Ok(chapters) => {
                                if chapters.iter().next().is_some() {
                                    acc.with_chapters.push(item);
                                } else {
                                    acc.without_chapters.push(item);
                                }
                            }
                            Err(e) => {
                                log::error!(
                                    "Chapter extract failed for {}: {e}",
                                    path_buf.display()
                                );
                                acc.skipped.push(item);
                            }
                        }
                        return acc;
                    }
                }
            }

            acc.skipped.push(item);
            acc
        })
        .reduce(ListDirSplit::default, |mut a, mut b| {
            a.with_chapters.append(&mut b.with_chapters);
            a.without_chapters.append(&mut b.without_chapters);
            a.skipped.append(&mut b.skipped);
            a
        });

    Ok(split)
}
