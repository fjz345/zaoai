#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::{ops::RangeInclusive, str::FromStr, time::Duration, thread::JoinHandle, sync::mpsc::Receiver};

use egui::plot::{Line, Plot, PlotPoints, PlotPoints::Owned};

use eframe::{
    egui::{self, plot::{PlotPoint, GridMark, GridInput}, style::Widgets, Response, Slider, RawInput},
    epaint::{Color32, Pos2, Rect},
    App, glow::TESS_EVALUATION_TEXTURE,
};
use graphviz_rust::{dot_structures::Graph, print};
use ndarray::{ArrayBase, OwnedRepr, Dim, Array2};
use symphonia::core::conv::IntoSample;

use crate::{
    egui_ext::{Interval, add_slider_sized},
    zneural_network::{
        datapoint::{split_datapoints, DataPoint, create_2x2_test_datapoints},
        neuralnetwork::{GraphStructure, NeuralNetwork, TrainingSession, TrainingState, AIResultMetadata, TrainingThreadPayload, TrainingThread},
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

    fn update_plot_data(&mut self, new_data: &Vec<PlotPoint>)
    {
        self.plot_data = new_data.clone();
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
                
                Self::create_plot_training().show(ui, |plot_ui| plot_ui.line(line));
            });
    }

    fn create_plot_training() -> Plot
    {
        const INCLUDE_Y_PADDING: f64 = 0.06;
        Plot::new("my_plot")
            .allow_drag(false).allow_zoom(false).allow_scroll(false).allow_boxed_zoom(false).allow_double_click_reset(false)
            .auto_bounds_x().center_x_axis(false).sharp_grid_lines(true)
            .include_y(0.0-INCLUDE_Y_PADDING).include_y(1.0 + INCLUDE_Y_PADDING).include_x(0.0)
            .y_grid_spacer(Self::create_plot_training_y_spacer_func)
            .width(500.0).height(300.0)
    }

    fn create_plot_training_y_spacer_func(grid: GridInput) -> Vec<GridMark>
    {
        vec![
        // 0.1
        GridMark { value: 0.05, step_size:  0.05 },
        GridMark { value: 0.1, step_size:  0.05 },
        GridMark { value: 0.15, step_size:  0.05 },
        GridMark { value: 0.2, step_size:  0.05 },
        GridMark { value: 0.25, step_size:  0.05 },
        GridMark { value: 0.3, step_size:  0.05 },
        GridMark { value: 0.35, step_size:  0.05 },
        GridMark { value: 0.4, step_size:  0.05 },
        GridMark { value: 0.45, step_size:  0.05 },
        GridMark { value: 0.6, step_size:  0.05 },
        GridMark { value: 0.65, step_size:  0.05 },
        GridMark { value: 0.7, step_size:  0.05 },
        GridMark { value: 0.75, step_size:  0.05 },
        GridMark { value: 0.8, step_size:  0.05 },
        GridMark { value: 0.85, step_size:  0.05 },
        GridMark { value: 0.9, step_size:  0.05 },
        GridMark { value: 0.95, step_size:  0.05 },

        // 0.25
        GridMark { value: 0.0, step_size: 0.25 },
        GridMark { value: 0.25, step_size: 0.25 },
        GridMark { value: 0.50, step_size: 0.25 },
        GridMark { value: 0.75, step_size: 0.25 },

        // 1.0
        GridMark { value: 0.0, step_size: 1.0 },
        GridMark { value: 1.0, step_size: 1.0 },
        ]
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

    pub fn get_training_data_slice(&self) -> &[DataPoint]
    {
        if self.is_split {
            return &self.training_split[..];
        } else {
            return &self.full_dataset[..];
        }
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
    training_session: TrainingSession,
    training_graph: TrainingGraphVisualization,
    training_thread: Option<TrainingThread>,
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
            training_session: TrainingSession::default(),
            training_graph: TrainingGraphVisualization::new(),
            training_thread: None,
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
    fn update_training_session(&mut self)
    {
        let mut ai_ref: Option<&NeuralNetwork> = self.ai.as_ref();
        self.training_session = TrainingSession::new(ai_ref,
             &self.training_dataset.get_training_data_slice(), 
             self.window_data.training_session_num_epochs, 
             self.window_data.training_session_batch_size, 
             self.window_data.training_session_learn_rate);
    }

    fn startup(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut visuals: egui::Visuals = egui::Visuals::dark();
        // visuals.panel_fill = Color32::from_rgba_unmultiplied(24, 36, 41, 255);
        ctx.set_visuals(visuals);
    }

    fn setup_ai(&mut self, nn_structure: GraphStructure) {
        self.training_session = TrainingSession::default();
        if (nn_structure.validate()) {
            self.ai = Some(NeuralNetwork::new(nn_structure));
            self.training_session.set_nn(self.ai.as_ref().unwrap());
        }
        
        self.window_data.training_session_num_epochs = self.training_session.get_num_epochs();
        self.window_data.training_session_batch_size = self.training_session.get_batch_size();
        self.window_data.training_session_learn_rate = self.training_session.get_learn_rate();
    }

    fn draw_ui_ai(&mut self, ctx: &egui::Context) {
        if !self.ai.is_some() {
            return;
        }

        let pos = egui::pos2(999999.0, 0.0);
        egui::Window::new("ZaoAI").default_pos(pos).show(ctx, |ui| {
            ui.label(self.ai.as_ref().unwrap().to_string());

            if ui.button("Test").clicked()
            {
                if self.ai.is_some()
                {
                    self.ai.as_mut().unwrap().test(&self.training_dataset.test_split[..]);
                }
            }
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
                    self.training_dataset.full_dataset.len(),
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
                let mut ui_dirty: bool = false;
                ui.horizontal(|ui| {
                    if add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.training_session_num_epochs,
                            RangeInclusive::new(1, 100),
                        )
                        .step_by(1.0),
                    ).changed() {
                        ui_dirty = true;
                    };
                    ui.label("Num Epochs");
                });

                ui.horizontal(|ui| {
                    if add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.training_session_batch_size,
                            RangeInclusive::new(10, 1000),
                        )
                        .step_by(10.0),
                    ).changed()
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
                            &mut self.window_data.training_session_learn_rate,
                            RangeInclusive::new(0.1, 0.5),
                        )
                        .step_by(0.1),
                    ).changed(){
                        ui_dirty = true;
                    };
                    ui.label("Learn Rate");
                });

                if ui_dirty
                {
                    self.update_training_session();
                }

                if ui.button("Begin Training").clicked() {
                    if self.training_session.ready() {
                        self.state = AppState::Training;
                        self.training_session.set_state(TrainingState::StartTraining);
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
            AppState::Idle => {self.draw_ui_menu(ctx, frame);}
            AppState::Training => {
                let training_state = self.training_session.get_state();
                match training_state
                {
                    TrainingState::Idle => {println!("TrainingState::Idle");}
                    TrainingState::StartTraining => {
                        if(self.training_thread.is_none())
                        {
                            // Copy the session for TrainingThread to take care of
                            self.training_thread = Some(TrainingThread::new(self.training_session.clone()));
                            self.training_session.set_state(TrainingState::Training);
                        }
                        else {
                            println!("Cannot start training when another one is in progress...");
                            self.training_session.set_state(TrainingState::Idle);
                        }
                    }
                    TrainingState::Training => {
                        let result_metadata = self.training_thread.as_ref().unwrap().rx_payload.try_recv();
                        let payload_buffer = &mut self.training_thread.as_mut().unwrap().payload_buffer;
                        if(result_metadata.is_ok())
                        {
                            payload_buffer.push(result_metadata.unwrap());

                            let training_plotpoints: Vec<PlotPoint> = generate_plotpoints_from_training_thread_payloads(&payload_buffer);
                            self.training_graph.update_plot_data(&training_plotpoints);
                        }

                        if payload_buffer.len() == payload_buffer.capacity()
                        {
                            self.training_session.set_state(TrainingState::Finish);
                        }
                    }
                    TrainingState::Finish => {
                        println!("Training Finished");

                        let result = self.training_thread.as_mut().unwrap().rx_neuralnetwork.try_recv();
                        if result.is_ok()
                        {
                            self.ai = Some(result.unwrap());
                        }
                        else {
                            panic!("Unexpected error");
                        }

                        self.training_thread.take().unwrap().handle.join();
                        self.training_thread = None;
                        self.training_session.set_state(TrainingState::Idle);
                        self.state = AppState::Idle;
                    }
                    TrainingState::Abort => {
                        panic!("Not Implemented");
                    }
                }

                self.draw_ui_menu(ctx, frame);
                ctx.request_repaint();
            }
            AppState::Exit => {
                frame.close();
            }

            default => {
                panic!("Not a valid state {:?}", self.state);
            }
        }
        // self.draw_ui_menu(ctx, frame);
    }

    fn post_rendering(&mut self, _window_size_px: [u32; 2], _frame: &eframe::Frame) {
        self.training_graph.should_show = self.window_data.show_training_graph;
    }
}

fn generate_plotpoints_from_training_thread_payloads(payloads: &Vec<TrainingThreadPayload>) -> Vec<PlotPoint>
{
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads
    {
        let asd = payload.training_metadata.calc_accuracy();
        let plotpoint = PlotPoint{ x: payload.payload_index as f64, y: asd };
        result.push(plotpoint);
    }
    result
}