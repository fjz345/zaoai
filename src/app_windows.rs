use std::ops::RangeInclusive;

use eframe::egui::{
    self,
    plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints}, Slider,
};

use crate::{app::{AppState, TrainingDataset}, egui_ext::{add_slider_sized, Interval}, mnist::get_mnist, zneural_network::{datapoint::create_2x2_test_datapoints, neuralnetwork::{NeuralNetwork, TrainingSession, TrainingState}}};

pub struct WindowTrainingGraph {
    pub(crate) title: String,
    pub(crate) should_show: bool,
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
        if !self.should_show {
            return ();
        }

        let pos = egui::pos2(999999.0, 999999.0);
        egui::Window::new(&self.title)
            .default_pos(pos)
            .show(ctx, |ui| {
                use crate::app_windows::PlotPoints::Owned;
                let plot_clone: PlotPoints = Owned(self.plot_data.clone());
                let line: Line = Line::new(plot_clone);

                Self::create_plot_training().show(ui, |plot_ui| plot_ui.line(line));
            });
    }

    fn create_plot_training() -> Plot {
        const INCLUDE_Y_PADDING: f64 = 0.06;
        Plot::new("my_plot")
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .auto_bounds_x()
            .center_x_axis(false)
            .sharp_grid_lines(true)
            .include_y(0.0 - INCLUDE_Y_PADDING)
            .include_y(1.0 + INCLUDE_Y_PADDING)
            .include_x(0.0)
            .y_grid_spacer(Self::create_plot_training_y_spacer_func)
            .width(500.0)
            .height(300.0)
    }

    fn create_plot_training_y_spacer_func(grid: GridInput) -> Vec<GridMark> {
        vec![
            // 0.1
            GridMark {
                value: 0.05,
                step_size: 0.05,
            },
            GridMark {
                value: 0.1,
                step_size: 0.05,
            },
            GridMark {
                value: 0.15,
                step_size: 0.05,
            },
            GridMark {
                value: 0.2,
                step_size: 0.05,
            },
            GridMark {
                value: 0.25,
                step_size: 0.05,
            },
            GridMark {
                value: 0.3,
                step_size: 0.05,
            },
            GridMark {
                value: 0.35,
                step_size: 0.05,
            },
            GridMark {
                value: 0.4,
                step_size: 0.05,
            },
            GridMark {
                value: 0.45,
                step_size: 0.05,
            },
            GridMark {
                value: 0.6,
                step_size: 0.05,
            },
            GridMark {
                value: 0.65,
                step_size: 0.05,
            },
            GridMark {
                value: 0.7,
                step_size: 0.05,
            },
            GridMark {
                value: 0.75,
                step_size: 0.05,
            },
            GridMark {
                value: 0.8,
                step_size: 0.05,
            },
            GridMark {
                value: 0.85,
                step_size: 0.05,
            },
            GridMark {
                value: 0.9,
                step_size: 0.05,
            },
            GridMark {
                value: 0.95,
                step_size: 0.05,
            },
            // 0.25
            GridMark {
                value: 0.0,
                step_size: 0.25,
            },
            GridMark {
                value: 0.25,
                step_size: 0.25,
            },
            GridMark {
                value: 0.50,
                step_size: 0.25,
            },
            GridMark {
                value: 0.75,
                step_size: 0.25,
            },
            // 1.0
            GridMark {
                value: 0.0,
                step_size: 1.0,
            },
            GridMark {
                value: 1.0,
                step_size: 1.0,
            },
        ]
    }
}

pub struct WindowAi {}

impl WindowAi {
    pub fn draw_ui(
        &self,
        ctx: &egui::Context,
        ai: Option<&mut NeuralNetwork>,
        test_training_dataset: &TrainingDataset,
    ) {
        if !ai.is_some() {
            return;
        }

        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            if let Some(ai) = ai {
                ui.label(ai.to_string());

                if ui.button("Test").clicked() {
                    ai.test(&test_training_dataset.test_split[..]);
                }
            } else {
                ui.label("NN not set");
            }
        });
    }
}


pub struct WindowTrainingSet
{

}

impl WindowTrainingSet
{
    pub fn draw_ui(&self, ctx: &egui::Context, training_dataset: &mut TrainingDataset, training_dataset_split_thresholds_0: &mut f64, training_dataset_split_thresholds_1: &mut f64) {
        // if !self.window_data.show_traning_dataset {
        //     return;
        // }

        let pos = egui::pos2(0.0, 600.0);
        egui::Window::new("Dataset")
            .default_pos(pos)
            .show(ctx, |ui: &mut egui::Ui| {
                ui.add(Interval::new(
                    training_dataset_split_thresholds_0,
                    training_dataset_split_thresholds_1,
                    RangeInclusive::new(0.0, 1.0),
                ));

                if ui.button("Split").clicked() {
                    training_dataset.split([
                        *training_dataset_split_thresholds_0,
                        *training_dataset_split_thresholds_1,
                    ]);
                }

                ui.heading("Current Dataset");
                ui.label(format!(
                    "Training: {} ({:.2}%)\nValidation: {} ({:.2}%)\nTest: {} ({:.2}%)\nTotal: {} ({:.2}%)",
                    training_dataset.training_split.len(),
                    training_dataset.thresholds[0],
                    training_dataset.validation_split.len(),
                    training_dataset.thresholds[1] - training_dataset.thresholds[0],
                    training_dataset.test_split.len(),
                    1.0 - training_dataset.thresholds[1],
                    training_dataset.full_dataset.len(),
                    (training_dataset.training_split.len()
                        + training_dataset.validation_split.len()
                        + training_dataset.test_split.len()) as f64
                        / training_dataset.full_dataset.len().max(1) as f64,
                ));

                ui.label(format!("Dimensions: ({}, {})", training_dataset.get_dimensions().0, training_dataset.get_dimensions().1));

                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    *training_dataset = TrainingDataset::new(&dataset);
                    training_dataset.split([1.0, 1.0]);
                }
                if ui.button("Load [784, 10] MNIST dataset").clicked()
                {
                    get_mnist();

                    // let mut dataset: Vec<DataPoint> = Vec::new();
                    // for (i, data) in train_data.iter().enumerate()
                    // {
                    //     dataset.push(DataPoint { inputs: [*train_data.get((i,0,0)).unwrap(), *train_data.get((i,0,1)).unwrap()], expected_outputs: [*train_labels.get((i,0)).unwrap(), *train_labels.get((i,0)).unwrap()] });
                    // }
                    
                    // training_dataset = TrainingDataset::new(&dataset);
                }
                
            });
    }
}

pub struct WindowTrainingSession
{

}
impl WindowTrainingSession
{
    pub fn draw_ui(&mut self, ctx: &egui::Context, training_session: &mut TrainingSession, app_state: &mut AppState) {
        // if !self.window_data.show_training_session {
        //     return;
        // }

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
                            RangeInclusive::new(0.1, 0.5),
                        )
                        .step_by(0.1),
                    )
                    .changed()
                    {
                        ui_dirty = true;
                    };
                    ui.label("Learn Rate");
                });

                if ui.button("Begin Training").clicked() {
                    if training_session.ready() {
                    *app_state = AppState::Training;
                        training_session
                            .set_state(TrainingState::StartTraining);
                    }
                }
            });
    }
}