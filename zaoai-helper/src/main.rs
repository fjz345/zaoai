use std::env;
use std::str::FromStr;

mod soloud;

use anyhow::Result;

use zaoai_types::spectrogram::{SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH};

use zaoai_types::ai_labels::{
    collect_zaoai_labels_multithread, generate_zaoai_label_spectrograms_multithread,
};
use zaoai_types::file::list_dir;
use zaoai_types::time::*;
use zaoai_types::utils::ListDirSplit;

use clap::Parser;
use zaoai_types::mkv::{collect_list_dir_split, path_exists};

#[derive(Parser)]
struct Args {
    #[arg(short, long, default_value = "")]
    media: String,
    #[arg(short, long, default_value = "")]
    output: String,

    // Generate listdirsplit
    #[arg(short, long, default_value_t = false)]
    listdirsplit: bool,
    // Generate .ZLBL(s) (Preq: listdirsplit)
    #[arg(short, long, default_value_t = false)]
    zlbl: bool,
    // Generate Spectrograms (Preq: ZLBL)
    #[arg(short, long, default_value_t = false)]
    spectrogram: bool,
}

fn resolve_str(env_var: &str, cli_arg: String, default: &str) -> String {
    std::env::var(env_var).unwrap_or_else(|_| {
        if !cli_arg.is_empty() {
            cli_arg
        } else {
            default.to_string()
        }
    })
}

fn resolve_parsed<T: FromStr>(env_var: &str, default: T) -> T {
    std::env::var(env_var)
        .ok()
        .and_then(|v| v.parse::<T>().ok())
        .unwrap_or(default)
}

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

    let media_path = resolve_str("ZAOAI_MEDIA_PATH", args.media, "test/test_Source");
    let output_path = resolve_str("OUTPUT_PATH", args.output, "output");

    if resolve_parsed("OUTPUT_PATH_CLEAR", false) {
        std::fs::remove_dir_all(&output_path)?;
    }
    std::fs::create_dir_all(&output_path)?;

    let zaoai_labels_out_path = format!("{output_path}/zaoai_labels");

    if args.listdirsplit {
        let _timer_scope = ScopeTimer::new("list_dir_split");
        let list_dir_split_out_path = format!("{output_path}/list_dir_split.json");

        path_exists(&media_path);

        let threads = resolve_parsed("LISTDIRSPLIT_NUM_THREADS", 0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        log::info!("listdirsplit threads: {}", pool.current_num_threads());
        pool.install(|| {
            if let Err(e) = collect_list_dir_split(media_path, list_dir_split_out_path) {
                log::error!("{}", e);
            }
        });
    }

    if args.zlbl {
        let _timer_scope = ScopeTimer::new("zaoai_labels");
        let read_list_dir_split =
            ListDirSplit::from_file_json("output/list_dir_split_001.json").unwrap();

        let threads = resolve_parsed("ZLBL_NUM_THREADS", 0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        log::info!("zlbl threads: {}", pool.current_num_threads());
        pool.install(|| {
            if let Err(e) =
                collect_zaoai_labels_multithread(&read_list_dir_split, &zaoai_labels_out_path)
            {
                log::error!("{}", e);
            }
        });
    }

    if args.spectrogram {
        let _timer_scope = ScopeTimer::new("spectrogram");

        let spectogram_width = resolve_parsed("SPECTROGRAM_WIDTH", SPECTROGRAM_WIDTH);
        let spectogram_height = resolve_parsed("SPECTROGRAM_HEIGHT", SPECTROGRAM_HEIGHT);
        let spectrogram_file_extension =
            resolve_str("SPECTROGRAM_EXTENSION", String::new(), "spectrogram");

        let list_dir = list_dir(zaoai_labels_out_path, true)?;

        let threads = resolve_parsed("SPECTROGRAM_NUM_THREADS", 0);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        log::info!("spectrogram threads: {}", pool.current_num_threads());
        pool.install(|| {
            if let Err(e) = generate_zaoai_label_spectrograms_multithread(
                &list_dir,
                &spectrogram_file_extension,
                [spectogram_width, spectogram_height],
            ) {
                log::error!("{}", e);
            };
        });
    }

    Ok(())
}
