use std::{env, fmt, time::Duration};

use clap::Parser;

#[derive(Parser)]
pub struct Args {
    #[arg(short, long, default_value = "")]
    pub media: String,
    #[arg(short, long, default_value = "")]
    pub output: String,
    #[arg(short, long, default_value_t = false)]
    pub delete_output: bool,

    #[arg(short, long, default_value_t = false)]
    pub listdirsplit: bool,
    #[arg(short, long, default_value_t = false)]
    pub zlbl: bool,
    #[arg(short, long, default_value_t = false)]
    pub spectrogram: bool,

    #[arg(long, default_value = "")]
    pub network_queue: String,
    #[arg(long, default_value = "")]
    pub network_workers: String,
    #[arg(long, default_value = "")]
    pub ffmpeg_queue: String,
    #[arg(long, default_value = "")]
    pub ffmpeg_workers: String,
    #[arg(long, default_value = "")]
    pub stall_timeout: String,
}

#[derive(Debug)]
pub struct ResolveError(String);

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ResolveError {}

pub trait ResolveValue: Sized {
    fn resolve_parse(value: &str) -> Result<Self, String>;
}

macro_rules! impl_resolve_value {
    ($($ty:ty),*) => {
        $(
            impl ResolveValue for $ty {
                fn resolve_parse(value: &str) -> Result<Self, String> {
                    value
                        .parse::<$ty>()
                        .map_err(|e| e.to_string())
                }
            }
        )*
    };
}

impl_resolve_value!(String, bool, i32, i64, f32, f64, usize);

impl ResolveValue for Duration {
    fn resolve_parse(value: &str) -> Result<Self, String> {
        let seconds = value.parse::<f64>().map_err(|e| e.to_string())?;

        Duration::try_from_secs_f64(seconds).map_err(|e| e.to_string())
    }
}

pub fn resolve<T>(cli_arg: String, env_var: &str, default: T) -> Result<T, ResolveError>
where
    T: ResolveValue,
{
    let parse = |value: &str, source: &str| {
        T::resolve_parse(value)
            .map_err(|e| ResolveError(format!("Invalid {source} value for {env_var}: {e}")))
    };

    if !cli_arg.is_empty() {
        parse(&cli_arg, "CLI")
    } else if let Ok(value) = env::var(env_var) {
        parse(&value, "environment")
    } else {
        Ok(default)
    }
}
