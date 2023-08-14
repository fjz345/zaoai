#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{ops::RangeInclusive, str::FromStr};

use egui::plot::{Line, Plot, PlotPoints, PlotPoints::Owned};

use eframe::{
    egui::{self, plot::PlotPoint, style::Widgets, Response, Slider},
    epaint::{Color32, Pos2, Rect},
    App,
};
use graphviz_rust::dot_structures::Graph;
use ndarray::{ArrayBase, OwnedRepr, Dim, Array2};

use crate::{
    egui_ext::Interval,
    zneural_network::{
        datapoint::{split_datapoints, DataPoint, create_2x2_test_datapoints},
        neuralnetwork::{GraphStructure, NeuralNetwork, TrainingSession, TrainingState},
    }, mnist::get_mnist,
};

struct TrainingGraphVisualization {
    title: String,
    should_show: bool,
    plot_data: Vec<PlotPoint>,
}

impl TrainingGraphVisualization {
    fn new() -> Self {
        let title = format!("Training Graph");
        Self {
            title,
            should_show: false,
            plot_data: Vec::new(),
        }
    }

    fn draw_ui(&mut self, ctx: &egui::Context) {
        if !self.should_show {
            return ();
        }

        let pos = egui::pos2(999999.0, 999999.0);
        egui::Window::new(&self.title)
            .default_pos(pos)
            .show(ctx, |ui| {
                let plot_clone: PlotPoints = Owned(self.plot_data.clone());
                let line: Line = Line::new(plot_clone);
                Plot::new("my_plot")
                    .view_aspect(2.0)
                    .width(500.0)
                    .height(300.0)
                    .show(ui, |plot_ui| plot_ui.line(line));
            });
    }
}

struct MenuWindowData {
    // Main Menu
    nn_structure: String,
    // Training Graph
    show_training_graph: bool,
    // Training Session
    show_training_session: bool,
    training_session_num_epochs: usize,
    training_session_batch_size: usize,
    training_session_learn_rate: f32,
    // Training Dataset
    show_traning_dataset: bool,
    training_dataset_split_thresholds_0: f64,
    training_dataset_split_thresholds_1: f64,
}

pub struct TrainingDataset {
    full_dataset: Vec<DataPoint>,
    is_split: bool,
    thresholds: [f64; 2],
    training_split: Vec<DataPoint>,
    validation_split: Vec<DataPoint>,
    test_split: Vec<DataPoint>,
}

impl TrainingDataset {
    pub fn new(datapoints: &[DataPoint]) -> Self {
        Self {
            full_dataset: datapoints.to_vec(),
            is_split: false,
            thresholds: [0.0; 2],
            training_split: Vec::new(),
            validation_split: Vec::new(),
            test_split: Vec::new(),
        }
    }

    pub fn split(&mut self, thresholds: [f64; 2]) {
        self.is_split = true;
        self.thresholds = thresholds;
        split_datapoints(
            &self.full_dataset[..],
            thresholds,
            self.training_split.as_mut(),
            self.validation_split.as_mut(),
            self.test_split.as_mut(),
        )
    }

    // Returns the number of (in, out) nodes needed in layers
    pub fn get_dimensions(&self) -> (usize, usize)
    {
        if(self.full_dataset.len() <= 0)
        {
            return (0,0);
        }

        (self.full_dataset[0].inputs.len(), self.full_dataset[0].expected_outputs.len())
    }
}

pub struct ZaoaiApp {
    state: AppState,
    ai: Option<NeuralNetwork>,
    window_data: MenuWindowData,
    training_dataset: TrainingDataset,
    training_session: Option<TrainingSession>,
    training_graph: TrainingGraphVisualization,
}

impl Default for ZaoaiApp {
    fn default() -> Self {
        Self {
            state: AppState::Startup,
            ai: None,
            window_data: MenuWindowData {
                nn_structure: "2, 3, 2".to_owned(),
                show_training_graph: true,
                show_training_session: true,
                training_session_num_epochs: 2,
                training_session_batch_size: 1000,
                training_session_learn_rate: 0.2,
                show_traning_dataset: true,
                training_dataset_split_thresholds_0: 0.75,
                training_dataset_split_thresholds_1: 0.9,
            },
            training_dataset: TrainingDataset::new(
                &[DataPoint {
                    inputs: [0.0; 2],
                    expected_outputs: [0.0; 2],
                }; 0],
            ),
            training_session: None,
            training_graph: TrainingGraphVisualization::new(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AppState {
    Startup,
    Idle,
    SetupAi,
    Training,
    Testing,
    Exit,
}

impl ZaoaiApp {
    fn startup(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut visuals: egui::Visuals = egui::Visuals::dark();
        // visuals.panel_fill = Color32::from_rgba_unmultiplied(24, 36, 41, 255);
        ctx.set_visuals(visuals);
    }

    fn setup_ai(&mut self, nn_structure: GraphStructure) {
        if (nn_structure.validate()) {
            self.ai = Some(NeuralNetwork::new(nn_structure));
        }
    }

    fn draw_ui_ai(&mut self, ctx: &egui::Context) {
        if !self.ai.is_some() {
            return;
        }

        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            ui.label(self.ai.as_ref().unwrap().to_string());
        });
    }

    fn draw_ui_training_dataset(&mut self, ctx: &egui::Context) {
        if !self.window_data.show_traning_dataset {
            return;
        }

        let pos = egui::pos2(0.0, 600.0);
        egui::Window::new("Dataset")
            .default_pos(pos)
            .show(ctx, |ui: &mut egui::Ui| {
                ui.add(Interval::new(
                    &mut self.window_data.training_dataset_split_thresholds_0,
                    &mut self.window_data.training_dataset_split_thresholds_1,
                    RangeInclusive::new(0.0, 1.0),
                ));

                if ui.button("Split").clicked() {
                    self.training_dataset.split([
                        self.window_data.training_dataset_split_thresholds_0,
                        self.window_data.training_dataset_split_thresholds_1,
                    ]);
                }

                ui.heading("Current Dataset");
                ui.label(format!(
                    "Training: {} ({:.2}%)\nValidation: {} ({:.2}%)\nTest: {} ({:.2}%)\nTotal: {} ({:.2}%)",
                    self.training_dataset.training_split.len(),
                    self.training_dataset.thresholds[0],
                    self.training_dataset.validation_split.len(),
                    self.training_dataset.thresholds[1] - self.training_dataset.thresholds[0],
                    self.training_dataset.test_split.len(),
                    1.0 - self.training_dataset.thresholds[1],
                    self.training_dataset.training_split.len()
                        + self.training_dataset.validation_split.len()
                        + self.training_dataset.test_split.len(),
                    (self.training_dataset.training_split.len()
                        + self.training_dataset.validation_split.len()
                        + self.training_dataset.test_split.len()) as f64
                        / self.training_dataset.full_dataset.len().max(1) as f64,
                ));

                ui.label(format!("Dimensions: ({}, {})", self.training_dataset.get_dimensions().0, self.training_dataset.get_dimensions().1));

                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    self.training_dataset = TrainingDataset::new(&dataset);
                    self.training_dataset.split([1.0, 1.0]);
                }
                if ui.button("Load [784, 10] MNIST dataset").clicked()
                {
                    get_mnist();

                    // let mut dataset: Vec<DataPoint> = Vec::new();
                    // for (i, data) in train_data.iter().enumerate()
                    // {
                    //     dataset.push(DataPoint { inputs: [*train_data.get((i,0,0)).unwrap(), *train_data.get((i,0,1)).unwrap()], expected_outputs: [*train_labels.get((i,0)).unwrap(), *train_labels.get((i,0)).unwrap()] });
                    // }
                    
                    // self.training_dataset = TrainingDataset::new(&dataset);
                }
                
            });
    }

    fn draw_ui_training_session(&mut self, ctx: &egui::Context) {
        if !self.window_data.show_training_session {
            return;
        }

        let pos = egui::pos2(500.0, 0.0);
        egui::Window::new("Training")
            .default_pos(pos)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.training_session_num_epochs,
                            RangeInclusive::new(1, 100),
                        )
                        .step_by(1.0),
                    );
                    ui.label("Num Epochs");
                });

                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.training_session_batch_size,
                            RangeInclusive::new(10, 1000),
                        )
                        .step_by(10.0),
                    );
                    ui.label("Batch Size");
                });

                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.training_session_learn_rate,
                            RangeInclusive::new(0.1, 0.5),
                        )
                        .step_by(0.1),
                    );
                    ui.label("Learn Rate");
                });

                if ui.button("Begin Training").clicked() {
                    if (self.training_session.is_some()) {
                        let mut training_session = self.training_session.as_mut().unwrap();

                        // Try to load traning dataset
                        if self.training_dataset.is_split {
                            training_session
                                .set_training_data(&self.training_dataset.training_split);
                        } else {
                            println!("Training with full dataset!");
                            training_session.set_training_data(&self.training_dataset.full_dataset);
                        }
                        self.state = AppState::Training;
                    }
                }
            });
    }

    fn draw_ui_menu(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.window_data.show_traning_dataset, "Show Dataset");

                ui.checkbox(&mut self.window_data.show_training_session, "Show Training");

                ui.checkbox(
                    &mut self.window_data.show_training_graph,
                    "Show Training Graph",
                );

                let name_label = ui.label("Create new NN with layers");
                if (ui
                    .text_edit_singleline(&mut self.window_data.nn_structure)
                    .labelled_by(name_label.id)
                    .lost_focus())
                {
                    self.state = AppState::SetupAi;
                }
            });
        });

        // Training Session
        self.draw_ui_training_session(ctx);

        self.draw_ui_training_dataset(ctx);

        self.draw_ui_ai(ctx);

        self.training_graph.draw_ui(ctx);
    }
}

impl eframe::App for ZaoaiApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match self.state {
            AppState::Startup => {
                self.startup(ctx, frame);
                self.state = AppState::SetupAi;
            }
            AppState::SetupAi => {
                let mut formatted_nn_structure = self
                    .window_data
                    .nn_structure
                    .split(|c| c == ',' || c == ' ')
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|str| -> usize {
                        let ret = FromStr::from_str(str).unwrap_or(0);
                        ret
                    })
                    .collect::<Vec<_>>();

                for i in (0..formatted_nn_structure.len()).rev() {
                    let nr = formatted_nn_structure[i];
                    if nr == 0 {
                        formatted_nn_structure.remove(i);
                    }
                }

                if (formatted_nn_structure.len() >= 2) {
                    self.setup_ai(GraphStructure::new(&formatted_nn_structure, true));
                }
                self.state = AppState::Idle;
            }
            AppState::Idle => {
                self.draw_ui_menu(ctx, frame);
            }
            AppState::Training => {
                if self.ai.is_none() {
                    self.state = AppState::Idle;
                    return;
                }

                let ai = self.ai.as_ref().unwrap();

                // Start Training
                if self.training_session.is_some() {
                    if self.training_session.as_ref().unwrap().ready() {
                        let mut training_session = self.training_session.as_ref().unwrap().clone();
                        let training_thread = training_session.begin();

                        while !training_thread.is_finished() {
                            self.draw_ui_menu(ctx, frame);
                        }

                        training_thread.join();
                    }
                }
            }
            AppState::Exit => {
                frame.close();
            }

            default => {
                panic!("Not a valid state {:?}", self.state);
            }
        }
    }

    fn post_rendering(&mut self, _window_size_px: [u32; 2], _frame: &eframe::Frame) {
        self.training_graph.should_show = self.window_data.show_training_graph;
    }
}

fn add_slider_sized(ui: &mut egui::Ui, size: f32, slider: egui::Slider) -> Response {
    let saved_slider_width = ui.style_mut().spacing.slider_width;
    ui.style_mut().spacing.slider_width = size;
    let result: Response = ui.add(slider);
    ui.style_mut().spacing.slider_width = saved_slider_width;
    result
}
