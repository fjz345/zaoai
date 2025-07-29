warning: unused import: `Chapters`
  --> E:\Github\zaoai-types\src\ai_labels.rs:19:16
   |
19 |     chapters::{Chapters, VideoMetadata},
   |                ^^^^^^^^
   |
   = note: `#[warn(unused_imports)]` on by default

warning: unused import: `anyhow::Context`
 --> E:\Github\zaoai-types\src\chapters.rs:3:5
  |
3 | use anyhow::Context;
  |     ^^^^^^^^^^^^^^^

warning: unused import: `OsStr`
 --> E:\Github\zaoai-types\src\chapters.rs:6:16
  |
6 | use std::ffi::{OsStr, OsString};
  |                ^^^^^

warning: unused import: `Display`
 --> E:\Github\zaoai-types\src\chapters.rs:7:16
  |
7 | use std::fmt::{Display, Write};
  |                ^^^^^^^

warning: unused import: `remove_file`
 --> E:\Github\zaoai-types\src\chapters.rs:8:21
  |
8 | use std::fs::{File, remove_file};
  |                     ^^^^^^^^^^^

warning: unused import: `std::io::Read`
 --> E:\Github\zaoai-types\src\chapters.rs:9:5
  |
9 | use std::io::Read;
  |     ^^^^^^^^^^^^^

warning: unused import: `PathBuf`
  --> E:\Github\zaoai-types\src\chapters.rs:10:23
   |
10 | use std::path::{Path, PathBuf};
   |                       ^^^^^^^

warning: unused import: `std::process::Stdio`
  --> E:\Github\zaoai-types\src\chapters.rs:11:5
   |
11 | use std::process::Stdio;
   |     ^^^^^^^^^^^^^^^^^^^

warning: unused import: `Component`
 --> E:\Github\zaoai-types\src\file.rs:3:12
  |
3 |     path::{Component, Path, PathBuf},
  |            ^^^^^^^^^

warning: unused import: `fs::File`
 --> E:\Github\zaoai-types\src\mkv.rs:2:5
  |
2 |     fs::File,
  |     ^^^^^^^^

warning: unused imports: `clear_folder_contents` and `self`
 --> E:\Github\zaoai-types\src\mkv.rs:9:12
  |
9 |     file::{self, clear_folder_contents, list_dir},
  |            ^^^^  ^^^^^^^^^^^^^^^^^^^^^

warning: unused import: `anyhow::Error`
  --> E:\Github\zaoai-types\src\mkv.rs:12:5
   |
12 | use anyhow::Error;
   |     ^^^^^^^^^^^^^

warning: unused imports: `Chapters` and `relative_path_from_base`
  --> E:\Github\zaoai-types\src\mkv.rs:18:23
   |
18 |     crate::chapters::{Chapters, extract_chapters},
   |                       ^^^^^^^^
19 |     crate::file::{EntryKind, relative_path_from_base},
   |                              ^^^^^^^^^^^^^^^^^^^^^^^

warning: unused import: `anyhow`
 --> E:\Github\zaoai-types\src\spectrogram.rs:1:31
  |
1 | use anyhow::{Context, Result, anyhow};
  |                               ^^^^^^

warning: unused import: `Read`
 --> E:\Github\zaoai-types\src\spectrogram.rs:4:10
  |
4 |     io::{Read, Write},
  |          ^^^^

warning: unused imports: `ColourGradient` and `SpecCompute`
 --> E:\Github\zaoai-types\src\spectrogram.rs:8:16
  |
8 | use sonogram::{ColourGradient, SonogramError, SpecCompute, SpecOptionsBuilder, Spectrogram};
  |                ^^^^^^^^^^^^^^                 ^^^^^^^^^^^

warning: unused imports: `Chapters`, `clear_folder_contents`, `self`, and `temp::copy_to_temp`
 --> E:\Github\zaoai-types\src\utils.rs:2:16
  |
2 |     chapters::{Chapters, extract_chapters},
  |                ^^^^^^^^
3 |     file::{self, EntryKind, clear_folder_contents, list_dir},
  |            ^^^^             ^^^^^^^^^^^^^^^^^^^^^
4 |     temp::copy_to_temp,
  |     ^^^^^^^^^^^^^^^^^^

warning: unused import: `Error`
 --> E:\Github\zaoai-types\src\utils.rs:6:23
  |
6 | use anyhow::{Context, Error, Result};
  |                       ^^^^^

warning: unused import: `File`
 --> E:\Github\zaoai-types\src\utils.rs:9:16
  |
9 |     fs::{self, File},
  |                ^^^^

warning: unused import: `std::io::Write`
  --> E:\Github\zaoai-types\src\utils.rs:14:5
   |
14 | use std::io::Write;
   |     ^^^^^^^^^^^^^^

warning: unused import: `time::Duration`
  --> E:\Github\zaoai-types\src\lib.rs:30:26
   |
30 | use std::{path::PathBuf, time::Duration};
   |                          ^^^^^^^^^^^^^^

warning: unused import: `VideoMetadata`
  --> E:\Github\zaoai-types\src\lib.rs:34:33
   |
34 | use crate::chapters::{Chapters, VideoMetadata};
   |                                 ^^^^^^^^^^^^^

warning: unreachable statement
  --> E:\Github\zaoai-types\src\ai_labels.rs:69:5
   |
67 |     return collect_zaoai_labels_multithread(list_dir_split, out_path);
   |     ----------------------------------------------------------------- any code following this expression is unreachable
68 |
69 |     let path_source = &list_dir_split.path_source.clone();
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unreachable statement
   |
   = note: `#[warn(unreachable_code)]` on by default

warning: unreachable statement
  --> E:\Github\zaoai-types\src\utils.rs:50:5
   |
48 |     return list_dir_with_kind_has_chapters_split_multithread(list, cull_empty_folders);
   |     ---------------------------------------------------------------------------------- any code following this expression is unreachable
49 |
50 |     let mut list_dir_split = ListDirSplit::default();
   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ unreachable statement

warning: unused variable: `temp_dir`
  --> E:\Github\zaoai-types\src\chapters.rs:18:10
   |
18 |     let (temp_dir, temp_file) = create_temp_file("chapters.xml")?;
   |          ^^^^^^^^ help: if this is intentional, prefix it with an underscore: `_temp_dir`
   |
   = note: `#[warn(unused_variables)]` on by default

warning: variable does not need to be mutable
   --> E:\Github\zaoai-types\src\chapters.rs:190:9
    |
190 |     let mut chapters = extract_chapters(mkv_file)?;
    |         ----^^^^^^^^
    |         |
    |         help: remove this `mut`
    |
    = note: `#[warn(unused_mut)]` on by default

warning: variable does not need to be mutable
   --> E:\Github\zaoai-types\src\mkv.rs:169:9
    |
169 |     let mut out_path = out_path.as_ref();
    |         ----^^^^^^^^
    |         |
    |         help: remove this `mut`

warning: unused variable: `v`
  --> E:\Github\zaoai-types\src\sound.rs:70:12
   |
70 |         Ok(v) => (),
   |            ^ help: if this is intentional, prefix it with an underscore: `_v`

warning: variable does not need to be mutable
  --> E:\Github\zaoai-types\src\spectrogram.rs:59:9
   |
59 |     let mut spectrogram = create_spectrogram_unsafe(buffer, width, height);
   |         ----^^^^^^^^^^^
   |         |
   |         help: remove this `mut`

warning: unused variable: `skipped`
  --> E:\Github\zaoai-types\src\utils.rs:86:29
   |
86 | ...                   skipped,
   |                       ^^^^^^^ help: try ignoring the field: `skipped: _`

warning: unused variable: `path_source`
  --> E:\Github\zaoai-types\src\utils.rs:87:29
   |
87 | ...                   path_source,
   |                       ^^^^^^^^^^^ help: try ignoring the field: `path_source: _`

warning: unused variable: `num_with_chapters`
  --> E:\Github\zaoai-types\src\utils.rs:88:29
   |
88 | ...                   num_with_chapters,
   |                       ^^^^^^^^^^^^^^^^^ help: try ignoring the field: `num_with_chapters: _`

warning: unused variable: `num_without_chapters`
  --> E:\Github\zaoai-types\src\utils.rs:89:29
   |
89 | ...                   num_without_chapters,
   |                       ^^^^^^^^^^^^^^^^^^^^ help: try ignoring the field: `num_without_chapters: _`

warning: unused variable: `num_skipped`
  --> E:\Github\zaoai-types\src\utils.rs:90:29
   |
90 | ...                   num_skipped,
   |                       ^^^^^^^^^^^ help: try ignoring the field: `num_skipped: _`

warning: fields `spec`, `width`, and `height` are never read
  --> E:\Github\zaoai-types\src\spectrogram.rs:78:9
   |
77 |     struct SpectrogramRepr {
   |            --------------- fields in this struct
78 |         spec: Vec<f32>,
   |         ^^^^
79 |         width: usize,
   |         ^^^^^
80 |         height: usize,
   |         ^^^^^^
   |
   = note: `#[warn(dead_code)]` on by default

warning: `zaoai-types` (lib) generated 35 warnings (run `cargo fix --lib -p zaoai-types` to apply 25 suggestions)
warning: lifetime flowing from input to output with different syntax can be confusing
   --> src\zneural_network\datapoint.rs:401:32
    |
401 |     pub fn training_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
    |                                ^^^^^                        ------------------------ the lifetime gets resolved as `'_`
    |                                |
    |                                this lifetime flows to the output
    |
    = note: `#[warn(mismatched_lifetime_syntaxes)]` on by default
help: one option is to remove the lifetime for references and use the anonymous lifetime for paths
    |
401 |     pub fn training_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter<'_> {
    |                                                                                     ++++

warning: lifetime flowing from input to output with different syntax can be confusing
   --> src\zneural_network\datapoint.rs:410:34
    |
410 |     pub fn validation_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
    |                                  ^^^^^                        ------------------------ the lifetime gets resolved as `'_`
    |                                  |
    |                                  this lifetime flows to the output
    |
help: one option is to remove the lifetime for references and use the anonymous lifetime for paths
    |
410 |     pub fn validation_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter<'_> {
    |                                                                                       ++++

warning: lifetime flowing from input to output with different syntax can be confusing
   --> src\zneural_network\datapoint.rs:419:28
    |
419 |     pub fn test_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter {
    |                            ^^^^^                        ------------------------ the lifetime gets resolved as `'_`
    |                            |
    |                            this lifetime flows to the output
    |
help: one option is to remove the lifetime for references and use the anonymous lifetime for paths
    |
419 |     pub fn test_batch_iter(&self, batch_size: usize) -> VirtualTrainingBatchIter<'_> {
    |                                                                                 ++++

warning: `zaoai` (bin "zaoai") generated 3 warnings
      Timing report saved to E:\Github\zaoai\target\cargo-timings\cargo-timing-20250729T190305.5023688Z.html
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
warning: the following packages contain code that will be rejected by a future version of Rust: svg v0.7.2
note: to see what the problems were, use the option `--future-incompat-report`, or run `cargo report future-incompatibilities --id 1`
