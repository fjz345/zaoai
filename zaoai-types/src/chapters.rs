#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use serde_xml_rs::de::from_str;
use std::ffi::OsString;
use std::fmt::Write;
use std::fs::File;
use std::path::Path;
use std::time::Duration;
use std::{fs, process::Command};

use crate::temp::create_temp_file;
use crate::utils::get_third_party_binary;
pub fn extract_chapters(mkv_file_path: impl AsRef<Path>) -> anyhow::Result<Option<Chapters>> {
    let (_temp_dir, temp_file) = create_temp_file("chapters.xml")?;

    println!(
        "Extracting chapters \"{}\" to \"{}\"",
        mkv_file_path.as_ref().display(),
        temp_file.display()
    );

    let tool_path = get_third_party_binary("mkvextract.exe");
    let mut command = Command::new(tool_path);
    let command = command
        .arg(mkv_file_path.as_ref())
        .arg("chapters")
        .arg(&temp_file);

    let output = command.output()?;
    let status = output.status;
    if !status.success() {
        println!("Input MKV path: {:?}", mkv_file_path.as_ref());
        println!("Output XML path: {:?}", temp_file);

        println!("stdout\n{:?}", String::from_utf8_lossy(&output.stdout));
        println!("stderr\n{:?}", String::from_utf8_lossy(&output.stderr));
        anyhow::bail!(
            "mkvextract failed on {} with status: {}",
            mkv_file_path.as_ref().display(),
            status
        );
    }

    let metadata = fs::metadata(&temp_file)?;
    if metadata.len() == 0 {
        return Ok(None); // no chapters found
    }

    let xml_content = fs::read_to_string(&temp_file)?;
    let chapters = parse_chapter_xml(&xml_content)?;

    Ok(Some(chapters))
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VideoMetadata {
    // Duration info
    #[serde(with = "humantime_serde")]
    pub duration: Duration,
    pub frame_count: Option<u32>,
    pub frame_rate: f32, // e.g., 23.976

    // Video stream info
    pub width: u32,
    pub height: u32,
    pub video_codec: Option<String>,

    // Audio info (optional)
    pub audio_codec: Option<String>,
    pub audio_language: Option<String>,
    pub audio_channels: Option<u8>,

    // Subtitle info (optional)
    pub subtitle_languages: Vec<String>,

    // Other
    pub container_format: Option<String>, // e.g. "mkv", "mp4"
    pub is_vfr: bool,                     // Variable Frame Rate?
    pub chapters: Vec<ChapterAtom>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl VideoMetadata {
    pub fn has_chapters(&self) -> bool {
        self.chapters.len() > 0
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct Chapters {
    #[serde(rename = "EditionEntry")]
    edition_entry: EditionEntry,
}

impl Into<Vec<ChapterAtom>> for Chapters {
    fn into(self) -> Vec<ChapterAtom> {
        let vec: Vec<ChapterAtom> = self.edition_entry.chapters.to_owned();
        vec
    }
}

impl Chapters {
    pub fn num_chapters(&self) -> usize {
        self.edition_entry.chapters.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &ChapterAtom> {
        self.edition_entry.chapters.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ChapterAtom> {
        self.edition_entry.chapters.iter_mut()
    }

    pub fn to_os_string(&self) -> OsString {
        let mut output = String::new();

        for chapter in self {
            let _ = writeln!(
                &mut output,
                "Start: {:<12} End: {:<12} Title: {}",
                chapter.start_time,
                chapter
                    .end_time
                    .clone()
                    .unwrap_or_else(|| "???".to_string()),
                chapter.display.title
            );
        }

        OsString::from(output)
    }
}

impl<'a> IntoIterator for &'a Chapters {
    type Item = &'a ChapterAtom;
    type IntoIter = std::slice::Iter<'a, ChapterAtom>;

    fn into_iter(self) -> Self::IntoIter {
        self.edition_entry.chapters.iter()
    }
}
impl<'a> IntoIterator for &'a mut Chapters {
    type Item = &'a mut ChapterAtom;
    type IntoIter = std::slice::IterMut<'a, ChapterAtom>;

    fn into_iter(self) -> Self::IntoIter {
        self.edition_entry.chapters.iter_mut()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct EditionEntry {
    #[serde(rename = "ChapterAtom", default)]
    chapters: Vec<ChapterAtom>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChapterAtom {
    #[serde(rename = "ChapterTimeStart")]
    pub start_time: String,

    #[serde(rename = "ChapterTimeEnd")]
    pub end_time: Option<String>,

    #[serde(rename = "ChapterDisplay")]
    pub display: ChapterDisplay,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChapterDisplay {
    #[serde(rename = "ChapterString")]
    pub title: String,
}

pub fn parse_chapter_xml(xml: &str) -> anyhow::Result<Chapters> {
    let chapters: Chapters = from_str(xml)?;
    Ok(chapters)
}

fn chapters_to_xml(chapters: &Chapters) -> anyhow::Result<String> {
    let inner = serde_xml_rs::to_string(chapters)?;
    Ok(format!(r#"<?xml version="1.0"?>\n{}"#, inner))
}

pub fn add_chapter_to_mkv(mkv_file: &str, timestamp: &str, title: &str) -> anyhow::Result<()> {
    let chapters = extract_chapters(mkv_file)?;
    if chapters.is_none() {
        println!("No chapters to add to mkv");
        return Ok(());
    }
    let mut chapters = chapters.unwrap();

    let new_chapter = ChapterAtom {
        start_time: timestamp.to_string(),
        end_time: None,
        display: ChapterDisplay {
            title: title.to_string(),
        },
    };
    chapters.edition_entry.chapters.push(new_chapter);

    let xml_output = chapters_to_xml(&chapters)?;

    use std::io::Write;
    let (_temp_dir, temp_file) = create_temp_file("add_chapter_to_mkv_chapters.xml")?;
    let mut file = File::create(&temp_file)?;
    file.write_all(xml_output.as_bytes())?;

    // Step 6: Apply changes via mkvpropedit
    let status = Command::new("third_party/bin/mkvpropedit.exe")
        .arg(mkv_file)
        .arg("--chapters")
        .arg(&temp_file)
        .status()?;

    if !status.success() {
        anyhow::bail!("Failed to apply chapters with mkvpropedit");
    }

    Ok(())
}
