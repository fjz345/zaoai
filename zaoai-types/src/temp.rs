use anyhow::{Context, Result};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

pub fn create_temp_file<P: AsRef<Path>>(source: P) -> Result<(TempDir, PathBuf)> {
    let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;
    let temp_path = temp_dir.path().join(
        source
            .as_ref()
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid source filename"))?,
    );

    fs::File::create(&temp_path)?;
    Ok((temp_dir, temp_path))
}

/// Recursively copies a file or directory to a temp directory.
/// Returns the path to the new copy.
pub fn copy_to_temp<P: AsRef<Path>>(source: P) -> Result<(TempDir, PathBuf)> {
    let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;
    let temp_path = temp_dir.path().join(
        source
            .as_ref()
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid source filename"))?,
    );

    if source.as_ref().is_file() {
        fs::copy(&source, &temp_path)
            .with_context(|| format!("Failed to copy file: {}", source.as_ref().display()))?;
    } else {
        copy_dir_recursive(&source.as_ref(), &temp_path)?;
    }

    Ok((temp_dir, temp_path))
}

/// Recursively copies a directory and its contents.
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else if file_type.is_file() {
            fs::copy(&src_path, &dst_path)
                .with_context(|| format!("Failed to copy file: {}", src_path.display()))?;
        }
    }
    Ok(())
}
