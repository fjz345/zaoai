use std::{cell::RefCell, ops::RangeInclusive, rc::Rc};

use crate::{
    app::{generate_plotpoints_from_training_thread_payloads, AppState, TrainingDataset},
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{create_2x2_test_datapoints, DataPoint},
        neuralnetwork::NeuralNetwork,
        thread::TrainingThread,
        training::{test_nn, TrainingSession, TrainingState},
    },
};
use eframe::egui::{self, Button, Sense, Slider};
use egui_plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints};
use serde::{Deserialize, Serialize};

pub trait DrawableWindow<'a> {
    type Ctx;

    fn with_ctx<F>(&mut self, egui_ctx: &egui::Context, ctx: &mut Self::Ctx, f: F)
    where
        F: FnOnce(&mut Self, &mut Self::Ctx),
    {
        f(self, ctx);
    }

    fn draw_ui(&mut self, ctx: &egui::Context, state_ctx: &mut Self::Ctx);
}

pub struct WindowTrainingGraphCtx<'a> {
    pub(crate) training_thread: &'a Option<TrainingThread>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct WindowTrainingGraph {
    #[serde(skip)]
    cached_plot_points: Vec<PlotPoint>,
}

impl<'a> DrawableWindow<'a> for WindowTrainingGraph {
    type Ctx = WindowTrainingGraphCtx<'a>;

    fn draw_ui(&mut self, ctx: &egui::Context, state_ctx: &mut Self::Ctx) {
        // Update
        if let Some(training_thread) = &state_ctx.training_thread {
            let payload_buffer = &state_ctx.training_thread.as_ref().unwrap().payload_buffer;
            let training_plotpoints =
                generate_plotpoints_from_training_thread_payloads(&payload_buffer);

            self.cached_plot_points = training_plotpoints;
        }

        egui::Window::new("Training Graph").show(ctx, |ui| {
            use crate::app_windows::PlotPoints::Owned;
            let plot_clone: PlotPoints = Owned(self.cached_plot_points.clone());
            let line: Line = Line::new("LineName", plot_clone);

            Self::create_plot_training().show(ui, |plot_ui| plot_ui.line(line));
        });
    }
}

impl WindowTrainingGraph {
    fn create_plot_training<'a>() -> Plot<'a> {
        const INCLUDE_Y_PADDING: f64 = 0.06;
        Plot::new("my_plot")
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .center_x_axis(false)
            .include_y(0.0 - INCLUDE_Y_PADDING)
            .include_y(1.0 + INCLUDE_Y_PADDING)
            .include_x(0.0)
            .y_grid_spacer(
                Self::create_plot_training_y_spacer_func as fn(GridInput) -> Vec<GridMark>,
            )
            .width(500.0)
            .height(300.0)
    }

    fn create_plot_training_y_spacer_func(grid: GridInput) -> Vec<GridMark> {
        let mut marks = Vec::new();

        // 0.05 step marks
        for &value in &[
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
            0.95,
        ] {
            marks.push(GridMark {
                value,
                step_size: 0.05,
            });
        }

        // 0.25 step marks
        for &value in &[0.0, 0.25, 0.5, 0.75] {
            marks.push(GridMark {
                value,
                step_size: 0.25,
            });
        }

        // 1.0 step marks
        for &value in &[0.0, 1.0] {
            marks.push(GridMark {
                value,
                step_size: 1.0,
            });
        }

        marks
    }
}

pub struct WindowAiCtx<'a> {
    pub ai: &'a mut Option<NeuralNetwork>,
    pub test_button_training_dataset: &'a Option<TrainingDataset>,
}

#[derive(Serialize, Deserialize)]
pub struct WindowAi {}

impl<'a> DrawableWindow<'a> for WindowAi {
    type Ctx = WindowAiCtx<'a>;

    fn draw_ui(&mut self, ctx: &egui::Context, state_ctx: &mut Self::Ctx) {
        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            if let Some(ai) = &mut state_ctx.ai {
                ui.label(ai.to_string());

                let sense = match state_ctx.test_button_training_dataset {
                    Some(_) => Sense::click(),
                    None => Sense::empty(),
                };
                let test_button = Button::new("Test").sense(sense);
                if ui.add(test_button).clicked() {
                    if let Some(training_dataset) = &state_ctx.test_button_training_dataset {
                        test_nn(ai, &training_dataset.test_split[..]);
                    } else {
                        log::error!("Training dataset not set, could not train");
                    }
                }
            } else {
                ui.label("NN not set");
            }
        });
    }
}

impl WindowAi {}

pub struct WindowTrainingSetCtx<'a> {
    pub training_dataset: &'a mut TrainingDataset, // Probably should store on heap to avoid copy, not an issue for now
}

#[derive(Serialize, Deserialize)]
pub struct WindowTrainingSet {
    ui_training_dataset_split_thresholds_0: f64,
    ui_training_dataset_split_thresholds_1: f64,
}

impl Default for WindowTrainingSet {
    fn default() -> Self {
        Self {
            ui_training_dataset_split_thresholds_0: 1.0,
            ui_training_dataset_split_thresholds_1: 1.0,
        }
    }
}

impl<'a> DrawableWindow<'a> for WindowTrainingSet {
    type Ctx = WindowTrainingSetCtx<'a>;

    fn draw_ui(&mut self, ctx: &egui::Context, state_ctx: &mut Self::Ctx) {
        egui::Window::new("Dataset")
            .default_pos([0.0, 600.0])
            .show(ctx, |ui| {
                               ui.add(Interval::new(
                    &mut self.ui_training_dataset_split_thresholds_0,
                    &mut self.ui_training_dataset_split_thresholds_1,
                    RangeInclusive::new(0.0, 1.0),
                ));

                if state_ctx.training_dataset.full_dataset.is_some()
                {
                    state_ctx.training_dataset.split([
                        self.ui_training_dataset_split_thresholds_0,
                        self.ui_training_dataset_split_thresholds_1,
                    ]);
                }
                    if let Some(full_dataset) = &state_ctx.training_dataset.full_dataset
                    {
              ui.heading("Current Dataset");
                ui.label(format!("Training: {} ({:.1}%)\nValidation: {} ({:.1}%)\nTest: {} ({:.1}%)\nTotal: {} ({:.1}%)",
                    state_ctx.training_dataset.training_split.len(),
                    100.0 * state_ctx.training_dataset.thresholds[0],
                    state_ctx.training_dataset.validation_split.len(),
                    100.0 * (state_ctx.training_dataset.thresholds[1] - state_ctx.training_dataset.thresholds[0]),
                    state_ctx.training_dataset.test_split.len(),
                    100.0 * (1.0 - state_ctx.training_dataset.thresholds[1]),
                    full_dataset.len(),
                    (state_ctx.training_dataset.training_split.len()
                        + state_ctx.training_dataset.validation_split.len()
                        + state_ctx.training_dataset.test_split.len()) as f64
                        / full_dataset.len().max(1) as f64,
                ));
                }
                ui.label(format!("Dimensions: ({}, {})", state_ctx.training_dataset.get_dimensions().0, state_ctx.training_dataset.get_dimensions().1));
                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    *state_ctx.training_dataset = TrainingDataset::new(&dataset);
                    state_ctx.training_dataset.split([1.0, 1.0]);
                }
                if ui.button("Load [784, 10] MNIST dataset").clicked()
                {
                    let mnist = get_mnist();
                    let dataset_train: Vec<DataPoint> = mnist.train_data.iter()
        .zip(mnist.train_labels.iter())
        .map(|(image, &label)| {
            // Normalize pixels to [0.0, 1.0]
            let inputs: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();

            let one_hot_encode = |label: u8, num_classes: usize| -> Vec<f32> {
                let mut v = vec![0.0; num_classes];
                if (label as usize) < num_classes {
                    v[label as usize] = 1.0;
                }
                v
            };
            let expected_outputs = one_hot_encode(label, 10);
            DataPoint { inputs, expected_outputs }
        })
        .collect();
        let dataset_test: Vec<DataPoint> = mnist.train_data.iter()
        .zip(mnist.train_labels.iter())
        .map(|(image, &label)| {
            // Normalize pixels to [0.0, 1.0]
            let inputs: Vec<f32> = image.iter().map(|&p| p as f32 / 255.0).collect();

            let one_hot_encode = |label: u8, num_classes: usize| -> Vec<f32> {
                let mut v = vec![0.0; num_classes];
                if (label as usize) < num_classes {
                    v[label as usize] = 1.0;
                }
                v
            };
            let expected_outputs = one_hot_encode(label, 10);
            DataPoint { inputs, expected_outputs }
        })
        .collect();
                    *state_ctx.training_dataset = TrainingDataset{ full_dataset: None, is_split: true, thresholds: [0.0,0.0], training_split: dataset_train, validation_split: vec![], test_split: dataset_test };
                }
            });
    }
}

pub struct WindowTrainingSessionCtx<'a> {
    pub training_session: &'a mut TrainingSession,
    pub app_state: &'a mut AppState,
    pub training_thread: &'a mut Option<TrainingThread>,
}

#[derive(Serialize, Deserialize)]
pub struct WindowTrainingSession {}

impl<'a> DrawableWindow<'a> for WindowTrainingSession {
    type Ctx = WindowTrainingSessionCtx<'a>;

    fn draw_ui(&mut self, ctx: &egui::Context, state_ctx: &mut Self::Ctx) {
        let pos = egui::pos2(500.0, 0.0);
        egui::Window::new("Training")
            .default_pos(pos)
            .show(ctx, |ui| {
                let mut ui_dirty: bool = false;
                ui.horizontal(|ui| {
                    if add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.num_epochs,
                            RangeInclusive::new(1, 100),
                        )
                        .clamping(egui::SliderClamping::Never)
                        .step_by(1.0),
                    )
                    .changed()
                    {
                        ui_dirty = true;
                    };
                    ui.label("Num Epochs");
                });

                ui.horizontal(|ui| {
                    if add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.batch_size,
                            RangeInclusive::new(10, 1000),
                        )
                        .clamping(egui::SliderClamping::Never)
                        .step_by(10.0),
                    )
                    .changed()
                    {
                        ui_dirty = true;
                    };
                    ui.label("Batch Size");
                });

                ui.horizontal(|ui| {
                    if add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.learn_rate,
                            RangeInclusive::new(0.01, 0.5),
                        )
                        .clamping(egui::SliderClamping::Never)
                        .min_decimals(2)
                        .max_decimals_opt(Some(5)),
                    )
                    .changed()
                    {
                        ui_dirty = true;
                    };
                    ui.label("Learn Rate");
                });

                if *state_ctx.app_state == AppState::Training {
                    if ui.button("Abort Training").clicked() {
                        log::info!("Training was interupted");
                        *state_ctx.training_thread = None;
                        *state_ctx.app_state = AppState::Idle;
                        state_ctx.training_session.set_state(TrainingState::Idle);
                    }
                } else {
                    if ui.button("Begin Training").clicked() {
                        if state_ctx.training_session.ready() {
                            *state_ctx.app_state = AppState::Training;
                            state_ctx
                                .training_session
                                .set_state(TrainingState::StartTraining);
                        } else {
                            log::error!(
                                "Could not start training, training_session not ready {:?}",
                                state_ctx.training_session
                            );
                        }
                    }
                }
            });
    }
}

impl WindowTrainingSession {}
