use crate::{
    chapters::extract_chapters,
    file::{EntryKind, list_dir},
};
use anyhow::{Context, Result};
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

pub fn list_dir_with_kind_has_chapters_split(
    list: &[EntryKind],
    cull_empty_folders: bool,
) -> Result<ListDirSplit> {
    // MULTIHREADED VERSION
    return list_dir_with_kind_has_chapters_split_multithread(list, cull_empty_folders);

    #[allow(unreachable_code)]
    let mut list_dir_split = ListDirSplit::default();

    for item in list {
        match item {
            EntryKind::File(path_buf) => {
                let mut has_chapters = false;
                if let Some(ext) = path_buf.extension() {
                    if ext == "mkv" {
                        let mkv_file_str = path_buf
                            .to_str()
                            .ok_or_else(|| anyhow::anyhow!("Invalid temp path string"))?;

                        // Read chapters from local copy
                        match extract_chapters(&mkv_file_str) {
                            Ok(chapters) => has_chapters = !chapters.iter().next().is_none(),
                            Err(e) => println!("{e}"),
                        }
                    }
                }

                if has_chapters {
                    list_dir_split.with_chapters.push(item.clone());
                } else {
                    list_dir_split.without_chapters.push(item.clone());
                }
            }
            EntryKind::Directory(path_buf) => {
                let path_buf_entry_list = list_dir(path_buf, cull_empty_folders)
                    .with_context(|| format!("Failed to copy file: {}", path_buf.display()))?;
                let dir_res =
                    list_dir_with_kind_has_chapters_split(&path_buf_entry_list, cull_empty_folders);
                match dir_res {
                    Ok(dir_split) => {
                        #[allow(unused_variables)]
                        let ListDirSplit {
                            with_chapters: directory_with_chapters,
                            without_chapters: directory_without_chapters,
                            skipped,
                            path_source,
                            num_with_chapters,
                            num_without_chapters,
                            num_skipped,
                        } = dir_split;
                        list_dir_split.with_chapters.extend(directory_with_chapters);
                        list_dir_split
                            .without_chapters
                            .extend(directory_without_chapters);
                    }
                    Err(e) => {
                        println!("{e}")
                    }
                }
            }
            EntryKind::Other(_) => {
                // Skip or handle Other if needed
            }
        }
    }

    Ok(list_dir_split)
}

pub fn list_dir_with_kind_has_chapters_split_multithread(
    list: &[EntryKind],
    cull_empty_folders: bool,
) -> Result<ListDirSplit> {
    let mut split = ListDirSplit::default();

    std::thread::scope(|scope| -> std::result::Result<(), anyhow::Error> {
        let mut handles = vec![];

        for item in list.iter().cloned() {
            let handle = scope.spawn(move || -> Result<ListDirSplit> {
                match &item {
                    EntryKind::File(path_buf) => {
                        let mut has_chapters = false;
                        if path_buf.extension().map_or(false, |ext| ext == "mkv") {
                            if let Some(mkv_file_str) = path_buf.to_str() {
                                match extract_chapters(mkv_file_str) {
                                    Ok(chapters) => has_chapters = chapters.iter().next().is_some(),
                                    Err(e) => eprintln!("Chapter extract failed: {e}"),
                                }
                            }
                        }

                        let mut s = ListDirSplit::default();
                        if has_chapters {
                            s.with_chapters.push(item);
                        } else {
                            s.without_chapters.push(item);
                        }
                        Ok(s)
                    }

                    EntryKind::Directory(path_buf) => {
                        let entries =
                            list_dir(path_buf, cull_empty_folders).with_context(|| {
                                format!("Failed to read directory: {}", path_buf.display())
                            })?;

                        // Recursive call — scoped
                        list_dir_with_kind_has_chapters_split_multithread(
                            &entries,
                            cull_empty_folders,
                        )
                    }

                    EntryKind::Other(_) => Ok(ListDirSplit::default()),
                }
            });

            handles.push(handle);
        }

        // Merge all thread results
        for handle in handles {
            match handle.join().expect("thread panicked") {
                Ok(local_split) => {
                    split.with_chapters.extend(local_split.with_chapters);
                    split.without_chapters.extend(local_split.without_chapters);
                }
                Err(e) => {
                    eprintln!("Worker failed: {e}");
                }
            }
        }

        Ok(())
    })?;

    Ok(split)
}
