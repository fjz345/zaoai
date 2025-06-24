use std::ops::RangeInclusive;

use crate::{
    app::{AppState, TrainingDataset},
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{create_2x2_test_datapoints, DataPoint},
        neuralnetwork::NeuralNetwork,
        thread::TrainingThread,
        training::{test_nn, TrainingSession, TrainingState},
    },
};
use eframe::egui::{self, Slider};
use egui_plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct WindowTrainingGraph {
    pub(crate) title: String,
    pub(crate) should_show: bool,
    #[serde(skip)]
    pub(crate) plot_data: Vec<PlotPoint>,
}

impl WindowTrainingGraph {
    pub fn new() -> Self {
        let title = format!("Training Graph");
        Self {
            title,
            should_show: false,
            plot_data: Vec::new(),
        }
    }

    pub fn update_plot_data(&mut self, new_data: &Vec<PlotPoint>) {
        self.plot_data = new_data.clone();
    }

    pub fn draw_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new(&self.title).show(ctx, |ui| {
            use crate::app_windows::PlotPoints::Owned;
            let plot_clone: PlotPoints = Owned(self.plot_data.clone());
            let line: Line = Line::new("LineName", plot_clone);

            Self::create_plot_training().show(ui, |plot_ui| plot_ui.line(line));
        });
    }

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

#[derive(Serialize, Deserialize)]
pub struct WindowAi {}
impl WindowAi {
    pub fn draw_ui(
        &self,
        ctx: &egui::Context,
        ai: Option<&mut NeuralNetwork>,
        test_training_dataset: &TrainingDataset,
    ) {
        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            if let Some(ai) = ai {
                ui.label(ai.to_string());

                if ui.button("Test").clicked() {
                    test_nn(ai, &test_training_dataset.test_split[..]);
                }
            } else {
                ui.label("NN not set");
            }
        });
    }
}

#[derive(Serialize, Deserialize)]
pub struct WindowTrainingSet {}

impl WindowTrainingSet {
    pub fn draw_ui(
        &self,
        ctx: &egui::Context,
        training_dataset: &mut TrainingDataset,
        training_dataset_split_thresholds_0: &mut f64,
        training_dataset_split_thresholds_1: &mut f64,
    ) {
        egui::Window::new("Dataset")
            .default_pos([0.0, 600.0])
            .show(ctx, |ui| {
                ui.add(Interval::new(
                    training_dataset_split_thresholds_0,
                    training_dataset_split_thresholds_1,
                    RangeInclusive::new(0.0, 1.0),
                ));

                if training_dataset.full_dataset.is_some()
                {
                    training_dataset.split([
                        *training_dataset_split_thresholds_0,
                        *training_dataset_split_thresholds_1,
                    ]);
                }
                    if let Some(full_dataset) = &training_dataset.full_dataset
                    {
              ui.heading("Current Dataset");
                ui.label(format!("Training: {} ({:.1}%)\nValidation: {} ({:.1}%)\nTest: {} ({:.1}%)\nTotal: {} ({:.1}%)",
                    training_dataset.training_split.len(),
                    100.0 * training_dataset.thresholds[0],
                    training_dataset.validation_split.len(),
                    100.0 * (training_dataset.thresholds[1] - training_dataset.thresholds[0]),
                    training_dataset.test_split.len(),
                    100.0 * (1.0 - training_dataset.thresholds[1]),
                    full_dataset.len(),
                    (training_dataset.training_split.len()
                        + training_dataset.validation_split.len()
                        + training_dataset.test_split.len()) as f64
                        / full_dataset.len().max(1) as f64,
                ));
                }
                ui.label(format!("Dimensions: ({}, {})", training_dataset.get_dimensions().0, training_dataset.get_dimensions().1));
                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    *training_dataset = TrainingDataset::new(&dataset);
                    training_dataset.split([1.0, 1.0]);
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
                    *training_dataset = TrainingDataset{ full_dataset: None, is_split: true, thresholds: [0.0,0.0], training_split: dataset_train, validation_split: vec![], test_split: dataset_test };
                }
            });
    }
}

#[derive(Serialize, Deserialize)]
pub struct WindowTrainingSession {}
impl WindowTrainingSession {
    pub fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        training_session: &mut TrainingSession,
        app_state: &mut AppState,
        training_thread: &mut Option<TrainingThread>,
    ) {
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
                            &mut training_session.num_epochs,
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
                            &mut training_session.batch_size,
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
                            &mut training_session.learn_rate,
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

                if *app_state == AppState::Training {
                    if ui.button("Abort Training").clicked() {
                        log::info!("Training was interupted");
                        *training_thread = None;
                        *app_state = AppState::Idle;
                        training_session.set_state(TrainingState::Idle);
                    }
                } else {
                    if ui.button("Begin Training").clicked() {
                        if training_session.ready() {
                            *app_state = AppState::Training;
                            training_session.set_state(TrainingState::StartTraining);
                        } else {
                            log::error!(
                                "Could not start training, training_session not ready {:?}",
                                training_session
                            );
                        }
                    }
                }
            });
    }
}
