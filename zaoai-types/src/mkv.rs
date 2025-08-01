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
    pub fn extract_opening_times(&self) -> (Option<Duration>, Option<Duration>) {
        let mut op_chapter = None;
        let mut op_chapter_next = None;
        let mut op_start = None;
        let mut op_end = None;

        // Not 100% optimal code, dont care
        for (i, chapter) in self.chapters.iter().enumerate() {
            let title = chapter.display.title.to_lowercase();
            if title.contains("op") || title.contains("opening") {
                op_chapter = Some(chapter);
                op_chapter_next = self.chapters.get(i + 1);
                op_start = parse_time(&chapter.start_time);
                op_end = chapter.end_time.as_ref().and_then(|s| parse_time(s));

                // Can finish early
                if op_start.is_some() && op_end.is_some() {
                    return (op_start, op_end);
                }
            }
        }

        if op_start.is_some() && op_chapter.is_some() {
            // If op_end not set, it means that we should use the next chapter start if it exists
            if let Some(next_chapter) = op_chapter_next {
                op_end = parse_time(&next_chapter.start_time);
            }
        }

        (op_start, op_end)
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
    // Ensure filename ends in "_XXX.json"
    if !ends_with_numbered_json(&out_path) {
        let file_stem = out_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid file_stem"))?;

        let dir = out_path.parent().unwrap_or_else(|| Path::new(""));
        let new_name = format!("{}_001.json", file_stem);
        out_path = dir.join(new_name);
    }

    // Extract base name (before _XXX), directory, and extension
    let dir = out_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .to_owned();
    let file_name = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;

    let re = regex::Regex::new(r"^(.*)_\d+\.json$")?;
    let caps = re
        .captures(file_name)
        .ok_or_else(|| anyhow::anyhow!("Filename not in expected format"))?;

    let base_name = &caps[1];

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

pub fn collect_list_dir_split(path: impl AsRef<Path>, out_path: impl AsRef<Path>) -> Result<()> {
    let out_path = out_path.as_ref();
    let list_of_entries = list_dir(&path, true).expect("");
    let mut list_dir_split =
        list_dir_with_kind_has_chapters_split(&list_of_entries, true).expect("");
    list_dir_split.path_source = path.as_ref().to_path_buf();
    list_dir_split.num_with_chapters = list_dir_split.with_chapters.len() as u32;
    list_dir_split.num_without_chapters = list_dir_split.without_chapters.len() as u32;
    list_dir_split.num_skipped = list_dir_split.skipped.len() as u32;

    let entry_kind_vec_string = |vec: &Vec<EntryKind>| -> Vec<String> {
        vec.iter()
            .map(|f| match f {
                EntryKind::File(path_buf)
                | EntryKind::Directory(path_buf)
                | EntryKind::Other(path_buf) => {
                    path_buf.file_stem().unwrap().to_string_lossy().to_string()
                }
            })
            .collect::<Vec<_>>()
    };
    println!(
        "With chapters[{}]: {:?}",
        &list_dir_split.num_with_chapters,
        entry_kind_vec_string(&list_dir_split.with_chapters)
    );
    println!(
        "Without chapters[{}]: {:?}",
        &list_dir_split.num_without_chapters,
        entry_kind_vec_string(&list_dir_split.without_chapters)
    );

    // Find a filename that does not exist (increase _00X+1)
    let next_file_name = find_next_available_file(out_path.to_path_buf())?;

    assert_eq!(ends_with_numbered_json(&next_file_name), true);
    let mut out_file = std::fs::File::create(&next_file_name)?;

    if !path_exists(&next_file_name) || !next_file_name.is_file() {
        anyhow::bail!("Not valid output file path");
    }
    out_file.write_all(&serde_json::to_vec_pretty(&list_dir_split)?)?;

    Ok(())
}

pub fn path_exists(path: impl AsRef<Path>) -> bool {
    let exists = std::path::Path::new(path.as_ref()).exists();
    if exists {
        println!("✅ Path exists: {}", path.as_ref().display());
    } else {
        println!("❌ Path does NOT exist: {}", path.as_ref().display());
    }

    exists
}
