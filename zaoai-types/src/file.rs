use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum EntryKind {
    File(PathBuf),
    Directory(PathBuf),
    #[allow(dead_code)]
    Other(PathBuf), // symlink, device, etc.
}

impl AsRef<PathBuf> for EntryKind {
    fn as_ref(&self) -> &PathBuf {
        match self {
            EntryKind::File(path_buf)
            | EntryKind::Directory(path_buf)
            | EntryKind::Other(path_buf) => return path_buf,
        }
    }
}

pub fn list_dir<P: AsRef<Path>>(path: P, cull_empty_folders: bool) -> Result<Vec<EntryKind>> {
    let path = path.as_ref();
    let entries = fs::read_dir(path)
        .with_context(|| format!("Failed to read directory: {}", path.display()))?;

    let mut results = Vec::new();
    for entry_result in entries {
        let entry = entry_result?;
        let path = entry.path();
        let metadata = entry.metadata()?;

        if metadata.is_file() {
            results.push(EntryKind::File(path));
        } else if metadata.is_dir() {
            if cull_empty_folders {
                let is_empty = fs::read_dir(&path)
                    .with_context(|| format!("Failed to read subdirectory: {}", path.display()))?
                    .next()
                    .is_none();
                if is_empty {
                    continue;
                }
            }
            results.push(EntryKind::Directory(path));
        } else {
            results.push(EntryKind::Other(path));
        }
    }

    Ok(results)
}

pub fn list_dir_all<P: AsRef<Path>>(path: P, cull_empty_folders: bool) -> Result<Vec<PathBuf>> {
    let list = list_dir(path, cull_empty_folders)?;

    let mut all_file_paths = Vec::new();
    for kind in list {
        match kind {
            EntryKind::File(path_buf) => all_file_paths.push(path_buf),
            EntryKind::Directory(path_buf) => {
                let res = list_dir_all(path_buf, cull_empty_folders)?;
                let collect_files: Vec<PathBuf> = res.iter().map(|f| f.clone()).collect();
                all_file_paths.extend(collect_files);
            }
            EntryKind::Other(_path_buf) => {}
        }
    }

    Ok(all_file_paths)
}

pub fn relative_path_from_base<'a>(file_path: &'a Path, base_dir: &'a Path) -> Result<&'a Path> {
    file_path.strip_prefix(base_dir).with_context(|| {
        format!(
            "Base directory '{}' is not a prefix of file path '{}'",
            base_dir.display(),
            file_path.display()
        )
    })
}

pub fn relative_after(path: &Path, base: &Path) -> Option<PathBuf> {
    path.strip_prefix(base).ok().map(|p| p.to_path_buf())
}

pub fn relative_before(path: &Path, base: &Path) -> Option<PathBuf> {
    let stripped = path.strip_prefix(base).ok()?;

    let mut components = stripped.components();
    let first_component = components.next()?;

    let mut new_path = PathBuf::from(base);
    new_path.push(first_component.as_os_str());

    Some(new_path)
}

pub fn clear_folder_contents(folder: &Path) -> std::io::Result<()> {
    // If is a directory
    if folder.is_dir() {
        for entry_result in fs::read_dir(folder)? {
            let entry = entry_result?;
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(&path)?; // recursively remove subfolder
            } else {
                fs::remove_file(&path)?; // remove file
            }
        }
    }
    Ok(())
}
