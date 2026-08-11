use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    chapters::{ChapterAtom, VideoMetadata},
    file::list_dir,
    utils::list_dir_with_kind_has_chapters_split,
};
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Write;
use {crate::chapters::extract_chapters, crate::file::EntryKind};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MkvMetadata {
    pub path: PathBuf,
    #[serde(with = "humantime_serde")]
    pub duration: Duration,
    pub chapters: Vec<ChapterAtom>,
}

impl Into<VideoMetadata> for MkvMetadata {
    fn into(self) -> VideoMetadata {
        VideoMetadata {
            container_format: Some("mkv".to_owned()),
            duration: self.duration,
            chapters: self.chapters,

            ..Default::default()
        }
    }
}

impl MkvMetadata {
    pub fn extract_opening_and_ending_times(
        &self,
    ) -> (
        (Option<Duration>, Option<Duration>),
        (Option<Duration>, Option<Duration>),
    ) {
        (
            self.find_chapter_times(&["op", "opening", "ncop"]),
            self.find_chapter_times(&["ed", "ending", "nced"]),
        )
    }

    fn find_chapter_times(&self, keywords: &[&str]) -> (Option<Duration>, Option<Duration>) {
        for (i, chapter) in self.chapters.iter().enumerate() {
            let title = chapter.display.title.to_lowercase();

            let is_match = title
                .split(|c: char| !c.is_alphanumeric())
                .any(|word| keywords.contains(&word));

            if is_match {
                let start = parse_time(&chapter.start_time);
                let end = chapter
                    .end_time
                    .as_ref()
                    .and_then(|s| parse_time(s))
                    .or_else(|| {
                        self.chapters
                            .get(i + 1)
                            .and_then(|next| parse_time(&next.start_time))
                    });

                return (start, end);
            }
        }

        (None, None)
    }
}

fn parse_time(s: &str) -> Option<Duration> {
    // Expected format: "HH:MM:SS.nnnnnnnnn"
    let parts: Vec<&str> = s.split([':', '.']).collect();
    if parts.len() != 4 {
        return None;
    }

    let hours = parts[0].parse::<u64>().ok()?;
    let minutes = parts[1].parse::<u64>().ok()?;
    let seconds = parts[2].parse::<u64>().ok()?;
    let nanos = parts[3].parse::<u32>().ok()?;

    Some(Duration::new(hours * 3600 + minutes * 60 + seconds, nanos))
}

// ffprobe -select_streams v -show_frames -show_entries frame=pkt_pts_time -of csv input.mkv

pub fn process_mkv_file(entry: &EntryKind) -> Result<MkvMetadata> {
    // Only process files
    let path = match entry {
        EntryKind::File(p) => p,
        _ => return Err(anyhow::anyhow!("Only processes files")),
    };

    // Check if it's an .mkv file
    if path.extension().and_then(|s| s.to_str()).unwrap_or("") != "mkv" {
        let string = format!("Only .mkv supported for now, {}", path.display());
        return Err(anyhow::anyhow!(string));
    }

    // Read chapters
    let chapters = extract_chapters(path)?.unwrap_or_default();

    let metadata = MkvMetadata {
        path: path.clone(),
        chapters: chapters.into(),
        duration: Duration::new(1337, 0),
    };

    Ok(metadata)
}

fn ends_with_numbered_json(path: impl AsRef<Path>) -> bool {
    let re = Regex::new(r"_\d+\.json$").unwrap();
    match path.as_ref().file_name().and_then(|name| name.to_str()) {
        Some(file_name) => re.is_match(file_name),
        None => false,
    }
}

fn find_next_available_file(mut out_path: PathBuf) -> Result<PathBuf> {
    if !ends_with_numbered_json(&out_path) {
        let file_stem = out_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid file_stem"))?;

        let dir = out_path.parent().unwrap_or_else(|| Path::new(""));
        let new_name = format!("{}_001.json", file_stem);
        out_path = dir.join(new_name);
    }

    let dir = out_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_owned();

    let file_stem = out_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;

    let (base_name, suffix) = file_stem
        .rsplit_once('_')
        .ok_or_else(|| anyhow::anyhow!("Filename not in expected format"))?;

    if !suffix.chars().all(|c| c.is_ascii_digit()) {
        anyhow::bail!("Filename not in expected format");
    }

    // Try _001, _002, ..., until file doesn't exist
    for i in 1..999 {
        let new_name = format!("{}_{:03}.json", base_name, i);
        let candidate = dir.join(new_name);
        if !candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!("Ran out of numbers (this should be practically impossible)")
}

pub fn collect_list_dir_split(
    path: impl AsRef<Path>,
    out_path: impl AsRef<Path>,
    limit: usize,
    progress_visualize_interval: Duration,
) -> Result<()> {
    let path = path.as_ref();
    let out_path = out_path.as_ref();

    let list_of_entries = list_dir(path, true)?;
    let mut list_dir_split = list_dir_with_kind_has_chapters_split(
        &list_of_entries,
        true,
        limit,
        progress_visualize_interval,
    )?;

    list_dir_split.path_source = path.to_path_buf();
    list_dir_split.num_with_chapters = list_dir_split.with_chapters.len() as u32;
    list_dir_split.num_without_chapters = list_dir_split.without_chapters.len() as u32;
    list_dir_split.num_skipped = list_dir_split.skipped.len() as u32;

    let next_file_name = find_next_available_file(out_path.to_path_buf())?;
    assert!(ends_with_numbered_json(&next_file_name));

    let mut out_file = std::fs::File::create(&next_file_name)?;

    if !path_exists(&next_file_name) || !next_file_name.is_file() {
        anyhow::bail!("Not valid output file path");
    }

    out_file.write_all(&serde_json::to_vec_pretty(&list_dir_split)?)?;

    log::trace!(
        "ListDirSplit:\n\
     ├─ with chapters:    {}\n\
     ├─ without chapters: {}\n\
     └─ skipped:          {}",
        list_dir_split.num_with_chapters,
        list_dir_split.num_without_chapters,
        list_dir_split.num_skipped,
    );

    Ok(())
}

pub fn path_exists(path: impl AsRef<Path>) -> bool {
    let exists = std::path::Path::new(path.as_ref()).exists();
    if exists {
        log::info!("✅ Path exists: {}", path.as_ref().display());
    } else {
        log::info!("❌ Path does NOT exist: {}", path.as_ref().display());
    }

    exists
}
