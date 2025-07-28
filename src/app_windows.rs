use std::{cell::RefCell, ops::RangeInclusive, path::PathBuf, rc::Rc};

use crate::{
    app::AppState,
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{
            create_2x2_test_datapoints, generate_spectogram, AnimeDataPoint, DataPoint,
            TrainingData, TrainingDataset, VirtualTrainingDataset,
        },
        neuralnetwork::NeuralNetwork,
        thread::{TrainingThread, TrainingThreadPayload},
        training::{test_nn, TrainingSession, TrainingState},
    },
};
use eframe::egui::{self, Button, Color32, InnerResponse, Response, Sense, Slider};
use egui_plot::{Corner, Legend};
use egui_plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints};
use serde::{Deserialize, Serialize};
use zaoai_types::ai_labels::ZaoaiLabelsLoader;

pub trait DrawableWindow<'a> {
    type Ctx;

    fn with_ctx<F>(&mut self, egui_ctx: &egui::Context, ctx: &mut Self::Ctx, f: F)
    where
        F: FnOnce(&mut Self, &mut Self::Ctx),
    {
        f(self, ctx);
    }

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>>;
}

pub struct WindowTrainingGraphCtx<'a> {
    pub(crate) training_thread: &'a Option<TrainingThread>,
}

#[derive(Clone, Serialize, Deserialize)]
struct SerdePlotPoint {
    x: f64,
    y: f64,
}

impl From<PlotPoint> for SerdePlotPoint {
    fn from(p: PlotPoint) -> Self {
        SerdePlotPoint { x: p.x, y: p.y }
    }
}

impl From<SerdePlotPoint> for PlotPoint {
    fn from(p: SerdePlotPoint) -> Self {
        PlotPoint::new(p.x, p.y)
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct WindowTrainingGraph {
    cached_plot_points_accuracy: Vec<SerdePlotPoint>,
    cached_plot_points_cost: Vec<SerdePlotPoint>,
    cached_plot_points_last_loss: Vec<SerdePlotPoint>,
}

impl<'a> DrawableWindow<'a> for WindowTrainingGraph {
    type Ctx = WindowTrainingGraphCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        // Update
        if let Some(training_thread) = &state_ctx.training_thread {
            let payload_buffer = &state_ctx.training_thread.as_ref().unwrap().payload_buffer;

            self.cached_plot_points_accuracy =
                generate_accuracy_plotpoints_from_training_thread_payloads(&payload_buffer)
                    .into_iter()
                    .map(|f| f.into())
                    .collect();
            self.cached_plot_points_cost =
                generate_cost_plotpoints_from_training_thread_payloads(&payload_buffer)
                    .into_iter()
                    .map(|f| f.into())
                    .collect();
            self.cached_plot_points_last_loss =
                generate_loss_plotpoints_from_training_thread_payloads(&payload_buffer)
                    .into_iter()
                    .map(|f| f.into())
                    .collect();
        }

        egui::Window::new("Training Graph").show(ctx, |ui| {
            use crate::app_windows::PlotPoints::Owned;

            let plot_accuracy: PlotPoints = Owned(
                self.cached_plot_points_accuracy
                    .clone()
                    .into_iter()
                    .map(|f| f.into())
                    .collect(),
            );
            let plot_cost: PlotPoints = Owned(
                self.cached_plot_points_cost
                    .clone()
                    .into_iter()
                    .map(|f| f.into())
                    .collect(),
            );
            let plot_loss: PlotPoints = Owned(
                self.cached_plot_points_last_loss
                    .clone()
                    .into_iter()
                    .map(|f| f.into())
                    .collect(),
            );

            let plot_cost_percent_first =
                *(plot_cost.points().first()).unwrap_or(&PlotPoint::new(0.0, 1.0));
            let plot_cost_percent = PlotPoints::new(
                plot_cost
                    .points()
                    .iter()
                    .map(|f| {
                        [
                            f.x,
                            f.y / plot_cost
                                .points()
                                .first()
                                .unwrap_or(&PlotPoint::new(0.0, 1.0))
                                .y,
                        ]
                    })
                    .collect(),
            );
            let plot_loss_percent_first =
                *(plot_loss.points().first()).unwrap_or(&PlotPoint::new(0.0, 1.0));
            let plot_loss_percent = PlotPoints::new(
                plot_loss
                    .points()
                    .iter()
                    .map(|f| [f.x, f.y / plot_loss_percent_first.y])
                    .collect(),
            );

            let line_accuracy = Line::new("Accuracy %", plot_accuracy).color(Color32::LIGHT_GREEN);
            let line_cost = Line::new(
                format!(
                    "Cost {:.0}",
                    (*plot_cost
                        .points()
                        .last()
                        .unwrap_or(&PlotPoint { x: 0.0, y: 0.0 }))
                    .y
                ),
                plot_cost,
            )
            .color(Color32::LIGHT_RED);
            let line_loss = Line::new(
                format!(
                    "Loss {:.0}",
                    (*plot_loss
                        .points()
                        .last()
                        .unwrap_or(&PlotPoint { x: 0.0, y: 0.0 }))
                    .y
                ),
                plot_loss,
            )
            .color(Color32::LIGHT_YELLOW);
            let line_cost_percent =
                Line::new("Cost %", plot_cost_percent).color(Color32::LIGHT_RED);
            let line_loss_percent =
                Line::new("Loss %", plot_loss_percent).color(Color32::LIGHT_YELLOW);

            // Create the plot once and add multiple lines inside it
            Self::create_plot_training()
                .legend(Legend::default().position(Corner::LeftTop))
                .x_axis_label("Epoch")
                .include_x(0.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(line_accuracy);
                    plot_ui.line(line_cost);
                    plot_ui.line(line_loss);
                    plot_ui.line(line_cost_percent);
                    plot_ui.line(line_loss_percent);
                });
        })
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
            .default_y_bounds(0.0 - INCLUDE_Y_PADDING, 1.0 + INCLUDE_Y_PADDING)
            .auto_bounds([true, false])
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
    pub test_button_training_data: &'a Option<&'a TrainingData>,
}

#[derive(Serialize, Deserialize)]
pub struct WindowAi {}

impl<'a> DrawableWindow<'a> for WindowAi {
    type Ctx = WindowAiCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            if let Some(ai) = &mut state_ctx.ai {
                ui.label(ai.to_string());

                let sense = match state_ctx.test_button_training_data {
                    Some(_) => Sense::click(),
                    None => Sense::empty(),
                };
                let test_button = Button::new("Test").sense(sense);
                if ui.add(test_button).clicked() {
                    if let Some(training_data) = state_ctx.test_button_training_data {
                        if training_data.test_split().len() >= 1 {
                            test_nn(ai, &training_data.test_split());
                        } else {
                            log::error!(
                                "Could not start test, training data training len was empty"
                            );
                        }
                    } else {
                        log::error!("Training dataset not set, could not train");
                    }
                }
                let delete_button = Button::new("Delete").sense(sense);
                if ui.add(delete_button).clicked() {
                    *state_ctx.ai = None;
                }
            } else {
                ui.label("NN not set");
            }
        })
    }
}

impl WindowAi {}

pub struct WindowTrainingSetCtx<'a> {
    pub training_data: &'a mut TrainingData, // Probably should store on heap to avoid copy, not an issue for now
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

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        egui::Window::new("Dataset")
            .default_pos([0.0, 600.0])
            .show(ctx, |ui| {
                               ui.add(Interval::new(
                    &mut self.ui_training_dataset_split_thresholds_0,
                    &mut self.ui_training_dataset_split_thresholds_1,
                    RangeInclusive::new(0.0, 1.0),
                ));
                state_ctx.training_data.set_thresholds(self.ui_training_dataset_split_thresholds_0, self.ui_training_dataset_split_thresholds_1);

                ui.heading("Current Dataset");
                ui.label(format!("Training: {} ({:.1}%)\nValidation: {} ({:.1}%)\nTest: {} ({:.1}%)\nTotal: {} ({:.1}%)",
                state_ctx.training_data.training_split().len(),
                100.0 * state_ctx.training_data.get_thresholds()[0],
                state_ctx.training_data.validation_split().len(),
                100.0 * (state_ctx.training_data.get_thresholds()[1] - state_ctx.training_data.get_thresholds()[0]),
                state_ctx.training_data.test_split().len(),
                100.0 * (1.0 - state_ctx.training_data.get_thresholds()[1]),
                state_ctx.training_data.len(),
                (state_ctx.training_data.training_split().len()
                    + state_ctx.training_data.validation_split().len()
                    + state_ctx.training_data.test_split().len()) as f64
                    / state_ctx.training_data.len().max(1) as f64,
                ));

                ui.label(format!("Dimensions: ({}, {})", state_ctx.training_data.get_dimensions().0, state_ctx.training_data.get_dimensions().1));
                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new(&dataset));
                    self.ui_training_dataset_split_thresholds_0 = state_ctx.training_data.get_thresholds()[0];
                    self.ui_training_dataset_split_thresholds_1 = state_ctx.training_data.get_thresholds()[1];
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
                    *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new_from_splits(&dataset_train, &vec![], &dataset_test));
                    self.ui_training_dataset_split_thresholds_0 = state_ctx.training_data.get_thresholds()[0];
                    self.ui_training_dataset_split_thresholds_1 = state_ctx.training_data.get_thresholds()[1];
                }
                const SPECTOGRAM_WIDTH: usize = 512;
                const SPECTOGRAM_HEIGHT: usize = 512;
                if ui.button(format!("Load [{}, {}] spectogram test", SPECTOGRAM_WIDTH*SPECTOGRAM_HEIGHT, 2)).clicked()
                {
                    let path = "test_files/test0.mkv";
                    let spectogram = generate_spectogram(&PathBuf::from(path));
                    let new_point = AnimeDataPoint {
                        path: PathBuf::from(path),
                        spectogram,
                        expected_outputs: vec![0.08936, 0.1510],
                    };

                    let dataset: Vec<_> = vec![new_point.into_data_point(SPECTOGRAM_WIDTH, SPECTOGRAM_HEIGHT)];
                    *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new(&dataset));
                    state_ctx.training_data.set_thresholds(1.0, 1.0);
                }

                // Todo: avoid constructing this each frame
                let zaoai_label_loader = ZaoaiLabelsLoader::new("training_data\\firstoutputlabels\\zaoai_labels").expect("Zaoailablesloader::new");
                if ui.button(format!("Load [{}, {}] {} ZaoaiLabels", SPECTOGRAM_WIDTH*SPECTOGRAM_HEIGHT, 2, zaoai_label_loader.len)).clicked()
                {
                    let zaoai_labels = zaoai_label_loader.load_zaoai_labels().expect("failed to load zaoai_labels");
                    *state_ctx.training_data = TrainingData::Virtual(VirtualTrainingDataset{ virtual_dataset: zaoai_labels, thresholds: [1.0, 1.0] });
                }
            })
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

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
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
                        .step_by(1.0)
                        .clamping(egui::SliderClamping::Never),
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
                        .step_by(1.0)
                        .clamping(egui::SliderClamping::Never),
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
                        *state_ctx.app_state = AppState::Training;
                        state_ctx
                            .training_session
                            .set_state(TrainingState::StartTraining);
                    }
                }
            })
    }
}

impl WindowTrainingSession {}

pub fn generate_accuracy_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let accuracy = payload.training_metadata.calc_accuracy();
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: accuracy,
        };
        result.push(plotpoint);
    }
    result
}

pub fn generate_cost_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let cost = payload.training_metadata.cost;
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: cost,
        };
        result.push(plotpoint);
    }
    result
}
pub fn generate_loss_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let last_loss = payload.training_metadata.last_loss;
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: last_loss,
        };
        result.push(plotpoint);
    }
    result
}
