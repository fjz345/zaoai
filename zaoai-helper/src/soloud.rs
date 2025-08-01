#![allow(dead_code)]

pub static S_IS_DEBUG: i32 = 1;
pub fn init_soloud() -> Soloud {
    let sl = Soloud::default().expect("Soloud Init failed");

    if S_IS_DEBUG > 0 {
        sl.set_visualize_enable(true);
    }

    sl
}

use std::process::Command;

use plotlib::page::Page;
use plotlib::repr::{Histogram, HistogramBins};
use plotlib::view::ContinuousView;
use soloud::{Soloud, audio};

pub static S_HISTOGRAM_MAX_X: f64 = 0.5;
pub static S_HISTOGRAM_MAX_Y: f64 = 160.0;
pub fn sl_debug(sl: &Soloud) {
    let fft = sl.calc_fft();

    let mut vec64: Vec<f64> = Vec::new();
    vec64.reserve(fft.len());

    for f32_entry in fft {
        vec64.push(f32_entry as f64);
    }

    let histogram = Histogram::from_slice(&vec64, HistogramBins::Count(vec64.len()));
    let view = ContinuousView::new()
        .add(histogram)
        .x_range(0.0, S_HISTOGRAM_MAX_X)
        .y_range(0.0, S_HISTOGRAM_MAX_Y);

    if !Command::new("cmd")
        .arg("/C")
        .arg("cls")
        .status()
        .unwrap()
        .success()
    {
        panic!("cls failed....");
    }
    log::info!(
        "{}",
        Page::single(&view)
            .dimensions((60.0 * 4.7) as u32, 15)
            .to_text()
            .unwrap()
    );
}

pub fn preview_sound_file(wav: audio::Wav) {
    let sl: Soloud = init_soloud();

    sl.play(&wav); // calls to play are non-blocking, so we put the thread to sleep
    while sl.voice_count() > 0 {
        if S_IS_DEBUG > 0 {
            sl_debug(&sl);
        } else {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    }
}
