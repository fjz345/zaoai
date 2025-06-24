// hide console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    ops::RangeInclusive, str::FromStr, sync::mpsc::Receiver, thread::JoinHandle, time::Duration,
};

use egui::plot::{Line, Plot, PlotPoints, PlotPoints::Owned};

use eframe::{
    egui::{
        self,
        plot::{GridInput, GridMark, PlotPoint},
        style::Widgets,
        RawInput, Response, Slider,
    },
    epaint::{Color32, Pos2, Rect},
    glow::TESS_EVALUATION_TEXTURE,
    App,
};
use graphviz_rust::{dot_structures::Graph, print};
use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use symphonia::core::conv::IntoSample;

use crate::{
    app_windows::{WindowAi, WindowTrainingGraph, WindowTrainingSession, WindowTrainingSet},
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{create_2x2_test_datapoints, split_datapoints, DataPoint},
        neuralnetwork::{
            AIResultMetadata, GraphStructure, NeuralNetwork, TrainingSession, TrainingState,
            TrainingThread, TrainingThreadPayload,
        },
    },
};

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
    pub full_dataset: Vec<DataPoint>,
    pub is_split: bool,
    pub thresholds: [f64; 2],
    pub training_split: Vec<DataPoint>,
    pub validation_split: Vec<DataPoint>,
    pub test_split: Vec<DataPoint>,
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

    pub fn get_training_data_slice(&self) -> &[DataPoint] {
        if self.is_split {
            return &self.training_split[..];
        } else {
            return &self.full_dataset[..];
        }
    }

    // Returns the number of (in, out) nodes needed in layers
    pub fn get_dimensions(&self) -> (usize, usize) {
        if (self.full_dataset.len() <= 0) {
            return (0, 0);
        }

        (
            self.full_dataset[0].inputs.len(),
            self.full_dataset[0].expected_outputs.len(),
        )
    }
}

pub struct ZaoaiApp {
    state: AppState,
    ai: Option<NeuralNetwork>,
    window_data: MenuWindowData,
    training_dataset: TrainingDataset,
    training_session: TrainingSession,
    training_thread: Option<TrainingThread>,
    window_graph: WindowTrainingGraph,
    window_ai: WindowAi,
    window_training_set: WindowTrainingSet,
    window_training_session: WindowTrainingSession,
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
            window_graph: WindowTrainingGraph::new(),
            training_thread: None,
            window_ai: WindowAi {},
            window_training_set: WindowTrainingSet {},
            window_training_session: WindowTrainingSession {},
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AppState {
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
        self.training_session = TrainingSession::default();
        if (nn_structure.validate()) {
            self.ai = Some(NeuralNetwork::new(nn_structure));
            self.training_session.set_nn(self.ai.as_ref().unwrap());
        }

        self.window_data.training_session_num_epochs = self.training_session.get_num_epochs();
        self.window_data.training_session_batch_size = self.training_session.get_batch_size();
        self.window_data.training_session_learn_rate = self.training_session.get_learn_rate();
    }

    fn draw_ui(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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

        // Windows
        self.window_training_session
            .draw_ui(ctx, &mut self.training_session, &mut self.state);
        self.window_training_set.draw_ui(
            ctx,
            &mut self.training_dataset,
            &mut self.window_data.training_dataset_split_thresholds_0,
            &mut self.window_data.training_dataset_split_thresholds_1,
        );
        self.window_ai
            .draw_ui(ctx, self.ai.as_mut(), &self.training_dataset);
        self.window_graph.draw_ui(ctx);
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
                self.draw_ui(ctx, frame);
            }
            AppState::Training => {
                let training_state = self.training_session.get_state();
                match training_state {
                    TrainingState::Idle => {
                        log::info!("TrainingState::Idle");
                    }
                    TrainingState::StartTraining => {
                        if (self.training_thread.is_none()) {
                            // Copy the session for TrainingThread to take care of
                            self.training_thread =
                                Some(TrainingThread::new(self.training_session.clone()));
                            self.training_session.set_state(TrainingState::Training);
                        } else {
                            log::info!("Cannot start training when another one is in progress...");
                            self.training_session.set_state(TrainingState::Idle);
                        }
                    }
                    TrainingState::Training => {
                        let result_metadata =
                            self.training_thread.as_ref().unwrap().rx_payload.try_recv();
                        let payload_buffer =
                            &mut self.training_thread.as_mut().unwrap().payload_buffer;
                        if (result_metadata.is_ok()) {
                            payload_buffer.push(result_metadata.unwrap());

                            let training_plotpoints: Vec<PlotPoint> =
                                generate_plotpoints_from_training_thread_payloads(&payload_buffer);
                            self.window_graph.update_plot_data(&training_plotpoints);
                        }

                        if payload_buffer.len() == payload_buffer.capacity() {
                            self.training_session.set_state(TrainingState::Finish);
                        }
                    }
                    TrainingState::Finish => {
                        log::info!("Training Finished");

                        let result = self
                            .training_thread
                            .as_mut()
                            .unwrap()
                            .rx_neuralnetwork
                            .try_recv();
                        if result.is_ok() {
                            self.ai = Some(result.unwrap());
                        } else {
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

                self.draw_ui(ctx, frame);
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
        self.window_graph.should_show = self.window_data.show_training_graph;
    }
}

fn generate_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let asd = payload.training_metadata.calc_accuracy();
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: asd,
        };
        result.push(plotpoint);
    }
    result
}
