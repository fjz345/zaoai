use std::env;
use std::path::PathBuf;
use std::time::Duration;

mod args;
mod soloud;

use anyhow::Result;

use clap::Parser;
use zaoai_types::spectrogram::{SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH};

use zaoai_types::ai_labels::{
    PipelineConfig, collect_zaoai_labels_multithread,
    generate_zaoai_label_spectrograms_queued_multithread,
};
use zaoai_types::file::{EntryKind, list_dir};
use zaoai_types::time::*;
use zaoai_types::utils::{ListDirSplit, collect_flat_files};

use zaoai_types::mkv::{collect_list_dir_split, path_exists};

use crate::args::*;

fn main() -> Result<()> {
    let _program_timer = ScopeTimer::new("program_total");
    color_backtrace::install();
    unsafe { env::set_var("RUST_BACKTRACE", "1") };
    unsafe { env::set_var("RUST_LOG", "debug") };
    dotenvy::dotenv().ok();
    env_logger::init();
    let args = Args::parse();
    if !(args.listdirsplit || args.zlbl || args.spectrogram) {
        log::error!(
            r#"Will not do any work, specify combination of "--listdirsplit", "--zlbl", "--spectrogram"#
        );
    }

    let media_path: String = resolve(args.media, "ZAOAI_MEDIA_PATH", "test/test_Source".into())?;
    let output_path: String = resolve(args.output, "OUTPUT_PATH", "output".into())?;
    let output_delete = resolve(args.delete_output.to_string(), "OUTPUT_DELETE", false)?;
    if output_delete {
        log::debug!("delete_output: {}", output_delete);
    }
    let limit = resolve(args.limit.to_string(), "LIMIT", 0 as usize)?;
    if limit != 0 {
        log::debug!("limit: {}", limit);
    }
    std::fs::create_dir_all(&output_path)?;
    path_exists(&output_path);

    let zaoai_labels_out_path = PathBuf::from(format!("{output_path}/zaoai_labels"));
    if args.listdirsplit {
        let _timer_scope = ScopeTimer::new("list_dir_split");

        let listdirsplit_filename = "list_dir_split.json";
        if output_delete {
            let pathbuf = PathBuf::from(&output_path);
            let mut count_deleted = 0;
            for i in 0..=999 {
                let filename = format!("list_dir_split_{:03}.json", i);
                let file_path = pathbuf.join(filename);
                if file_path.exists() {
                    log::debug!("Removing file: {}", file_path.display());
                    std::fs::remove_file(file_path)?;
                    count_deleted += 1;
                }
            }
            log::debug!("Deleted list_dir_split_: {}", count_deleted);
        }

        let list_dir_split_out_path = format!("{output_path}/{listdirsplit_filename}");

        path_exists(&media_path);

        let threads: usize = resolve("".into(), "LISTDIRSPLIT_NUM_THREADS", 0)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        log::info!("listdirsplit threads: {}", pool.current_num_threads());
        pool.install(|| {
            if let Err(e) = collect_list_dir_split(media_path, list_dir_split_out_path, limit) {
                log::error!("{}", e);
            }
        });
    }

    if args.zlbl {
        let _timer_scope = ScopeTimer::new("zaoai_labels");

        std::fs::create_dir_all(&zaoai_labels_out_path)?;
        path_exists(&zaoai_labels_out_path);

        let read_list_dir_split =
            ListDirSplit::from_file_json("output/list_dir_split_001.json").unwrap();

        if output_delete {
            let mut flat_files = Vec::new();
            collect_flat_files(
                &[EntryKind::Directory(zaoai_labels_out_path.clone())],
                false,
                &mut flat_files,
                0,
            )?;
            let zlbl_extension = "zlbl";
            let mut count_deleted = 0;
            for item in flat_files {
                if let EntryKind::File(path) = item {
                    if path.extension().and_then(|ext| ext.to_str()) == Some(zlbl_extension) {
                        log::trace!("Deleting: {}", path.display());
                        std::fs::remove_file(path)?;
                        count_deleted += 1;
                    }
                }
            }
            log::debug!("Deleted .{}: {}", zlbl_extension, count_deleted);
        }

        let threads = resolve("".into(), "ZLBL_NUM_THREADS", 0)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        log::info!("zlbl threads: {}", pool.current_num_threads());
        pool.install(|| {
            if let Err(e) = collect_zaoai_labels_multithread(
                &read_list_dir_split,
                &zaoai_labels_out_path,
                limit,
            ) {
                log::error!("{}", e);
            }
        });
    }

    if args.spectrogram {
        let _timer_scope = ScopeTimer::new("spectrogram");

        std::fs::create_dir_all(&zaoai_labels_out_path)?;
        path_exists(&zaoai_labels_out_path);

        let spectogram_width = resolve("".into(), "SPECTROGRAM_WIDTH", SPECTROGRAM_WIDTH)?;
        let spectogram_height = resolve("".into(), "SPECTROGRAM_HEIGHT", SPECTROGRAM_HEIGHT)?;
        let spectrogram_file_extension: String =
            resolve("".into(), "SPECTROGRAM_EXTENSION", "spectrogram".into())?;

        if output_delete {
            let mut flat_files = Vec::new();
            collect_flat_files(
                &[EntryKind::Directory(zaoai_labels_out_path.clone())],
                false,
                &mut flat_files,
                0,
            )?;
            let mut count_deleted = 0;
            for item in flat_files {
                if let EntryKind::File(path) = item {
                    if path.extension().and_then(|ext| ext.to_str())
                        == Some(spectrogram_file_extension.as_str())
                    {
                        log::trace!("Deleting: {}", path.display());
                        std::fs::remove_file(path)?;
                        count_deleted += 1;
                    }
                }
            }
            log::debug!("Deleted .{}: {}", spectrogram_file_extension, count_deleted);
        }

        let list_dir = list_dir(zaoai_labels_out_path, true)?;

        const DEFAULT_QUEUE_SIZE_NETWORK: usize = 4;
        const DEFAULT_NETWORK_WORKERS: usize = 2;
        const DEFAULT_QUEUE_SIZE_FFMPEG: usize = 4;
        const DEFAULT_FFMPEG_WORKERS: usize = 8;
        const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(10);

        let temp_dir_string: String = resolve(args.temp_dir, "TEMP_DIR", "".into())?;
        let custom_temp_dir = if temp_dir_string != "" {
            Some(temp_dir_string.into())
        } else {
            None
        };
        if let Err(e) = generate_zaoai_label_spectrograms_queued_multithread(
            &list_dir,
            &spectrogram_file_extension,
            [spectogram_width, spectogram_height],
            custom_temp_dir,
            PipelineConfig {
                network_queue_size: resolve(
                    args.network_queue,
                    "SPECTROGRAM_NETWORK_QUEUE",
                    DEFAULT_QUEUE_SIZE_NETWORK,
                )?,
                network_workers: resolve(
                    args.network_workers,
                    "SPECTROGRAM_NETWORK_WORKERS",
                    DEFAULT_NETWORK_WORKERS,
                )?,
                ffmpeg_queue_size: resolve(
                    args.ffmpeg_queue,
                    "SPECTROGRAM_FFMPEG_QUEUE",
                    DEFAULT_QUEUE_SIZE_FFMPEG,
                )?,
                ffmpeg_workers: resolve(
                    args.ffmpeg_workers,
                    "SPECTROGRAM_FFMPEG_WORKERS",
                    DEFAULT_FFMPEG_WORKERS,
                )?,
                stall_timeout: resolve(args.stall_timeout, "STALL_TIMEOUT", DEFAULT_STALL_TIMEOUT)?,
            },
            limit,
        ) {
            log::error!("{}", e);
        };
    }

    Ok(())
}
