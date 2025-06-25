// hide console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use eframe::{
    egui::{self, style::Widgets, InnerResponse, RawInput, Response, Slider},
    epaint::{Color32, Pos2, Rect},
    glow::TESS_EVALUATION_TEXTURE,
    App,
};
use egui_plot::PlotPoint;
use graphviz_rust::{dot_structures::Graph, print};
use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::{
    ops::RangeInclusive, str::FromStr, sync::mpsc::Receiver, thread::JoinHandle, time::Duration,
};
use symphonia::core::conv::IntoSample;

use crate::{
    app_windows::{WindowAi, WindowTrainingGraph, WindowTrainingSession, WindowTrainingSet},
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{create_2x2_test_datapoints, split_datapoints, DataPoint},
        neuralnetwork::{GraphStructure, NeuralNetwork},
        thread::{TrainingThread, TrainingThreadPayload},
        training::{TrainingSession, TrainingState},
    },
};

#[derive(Serialize, Deserialize)]
struct MenuWindowData {
    // Main Menu
    nn_structure: String,
    show_ai: bool,
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

#[derive(Serialize, Deserialize, Default)]
pub struct TrainingDataset {
    pub full_dataset: Option<Vec<DataPoint>>,
    pub is_split: bool,
    pub thresholds: [f64; 2],
    pub training_split: Vec<DataPoint>,
    pub validation_split: Vec<DataPoint>,
    pub test_split: Vec<DataPoint>,
}

impl TrainingDataset {
    pub fn new(datapoints: &[DataPoint]) -> Self {
        Self {
            full_dataset: Some(datapoints.to_vec()),
            is_split: false,
            thresholds: [0.0; 2],
            training_split: Vec::new(),
            validation_split: Vec::new(),
            test_split: Vec::new(),
        }
    }

    pub fn split(&mut self, thresholds: [f64; 2]) {
        if let Some(full) = &self.full_dataset {
            self.is_split = true;
            self.thresholds = thresholds;
            split_datapoints(
                &full[..],
                thresholds,
                self.training_split.as_mut(),
                self.validation_split.as_mut(),
                self.test_split.as_mut(),
            )
        } else {
            log::error!("Tried to split win full_dataset = None");
        }
    }

    pub fn get_datapoint_iter(&self) -> impl Iterator<Item = &DataPoint> + '_ {
        self.training_split
            .iter()
            .chain(self.validation_split.iter())
            .chain(self.test_split.iter())
    }

    // Returns the number of (in, out) nodes needed in layers
    pub fn get_dimensions(&self) -> (usize, usize) {
        if let Some(full_dataset) = &self.full_dataset {
            if full_dataset.len() >= 1 {
                (
                    full_dataset[0].inputs.len(),
                    full_dataset[0].expected_outputs.len(),
                )
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ZaoaiApp {
    #[serde(skip)]
    state: AppState,
    ai: Option<NeuralNetwork>,
    window_data: MenuWindowData,
    #[serde(skip)]
    training_dataset: TrainingDataset,
    training_session: TrainingSession,
    #[serde(skip)]
    training_thread: Option<TrainingThread>,
    window_training_graph: WindowTrainingGraph,
    window_ai: WindowAi,
    window_training_set: WindowTrainingSet,
    window_training_session: WindowTrainingSession,
}

impl eframe::App for ZaoaiApp {
    #[cfg(not(feature = "linux-profile"))]
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        log::info!("SAVING...");

        #[cfg(feature = "serde")]
        if let Ok(json) = serde_json::to_string(self) {
            log::debug!("SAVED with state: {:?}", self.state);
            storage.set_string(eframe::APP_KEY, json);
        }
        log::info!("SAVED!");
    }

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
                    self.setup_ai(GraphStructure::new(&formatted_nn_structure));
                } else {
                    log::error!("SetupAI failed, formatted_nn_structure.len() < 2");
                }
                self.state = AppState::Idle;
            }
            AppState::Idle => {
                let response = self.draw_ui(ctx, frame);

                // ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(
                //     response.response.rect.size(),
                // ));
            }
            AppState::Training => {
                let training_state = self.training_session.get_state();
                match training_state {
                    TrainingState::Idle => {
                        log::trace!("TrainingState::Idle");
                    }
                    TrainingState::StartTraining => {
                        if let Some(ai) = &self.ai {
                            if (self.training_thread.is_none()) {
                                if let Some(first_point) =
                                    self.training_session.training_data.first()
                                {
                                    if first_point.inputs.len() == ai.graph_structure.input_nodes
                                        && first_point.expected_outputs.len()
                                            == ai.graph_structure.output_nodes
                                    {
                                        // Copy the session for TrainingThread to take care of
                                        self.training_thread = Some(TrainingThread::new(
                                            self.training_session.clone(),
                                        ));
                                        self.training_session.set_state(TrainingState::Training);
                                    } else {
                                        log::error!("Cannot start training, dimension missmatch (NN: {}/{}) != (DP: {}/{})", ai.graph_structure.input_nodes, ai.graph_structure.output_nodes, first_point.inputs.len(), first_point.expected_outputs.len());
                                        self.training_session.set_state(TrainingState::Idle);
                                    }
                                } else {
                                    log::error!("Cannot start training, datapoint len <= 0");
                                    self.training_session.set_state(TrainingState::Idle);
                                }
                            } else {
                                log::error!(
                                    "Cannot start training when another one is in progress..."
                                );
                                self.training_session.set_state(TrainingState::Training);
                            }
                        } else {
                            log::error!("Cannot start training, NN not set");
                            self.training_session.set_state(TrainingState::Idle);
                        }
                    }
                    TrainingState::Training => {
                        let result_metadata = self
                            .training_thread
                            .as_mut()
                            .unwrap()
                            .rx_payload
                            .as_mut()
                            .expect("ERROR")
                            .try_recv();
                        let payload_buffer =
                            &mut self.training_thread.as_mut().unwrap().payload_buffer;
                        if (result_metadata.is_ok()) {
                            payload_buffer.push(result_metadata.unwrap());

                            let training_plotpoints: Vec<PlotPoint> =
                                generate_plotpoints_from_training_thread_payloads(&payload_buffer);
                            self.window_training_graph
                                .update_plot_data(&training_plotpoints);
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
                            .as_mut()
                            .expect("ERROR")
                            .try_recv();
                        if result.is_ok() {
                            self.ai = Some(result.unwrap());
                        } else {
                            panic!("Unexpected error");
                        }

                        self.training_thread
                            .take()
                            .unwrap()
                            .handle
                            .expect("ERROR")
                            .join();
                        self.training_thread = None;
                        self.training_session.set_state(TrainingState::Idle);
                        self.state = AppState::Idle;
                    }
                    TrainingState::Abort => {
                        panic!("Not Implemented");
                    }
                }

                let response = self.draw_ui(ctx, frame);
                ctx.request_repaint();
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(
                    response.response.rect.size(),
                ));
            }
            AppState::Exit => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }

            default => {
                panic!("Not a valid state {:?}", self.state);
            }
        }
    }
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
                show_ai: true,
            },
            training_dataset: TrainingDataset::new(
                &[DataPoint {
                    inputs: vec![0.0; 2],
                    expected_outputs: vec![0.0; 2],
                }; 0],
            ),
            training_session: TrainingSession::default(),
            window_training_graph: WindowTrainingGraph::new(),
            training_thread: None,
            window_ai: WindowAi {},
            window_training_set: WindowTrainingSet {},
            window_training_session: WindowTrainingSession {},
        }
    }
}

#[derive(Default, Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AppState {
    #[default]
    Startup,
    Idle,
    SetupAi,
    Training,
    Testing,
    Exit,
}

impl ZaoaiApp {
    pub fn new(cc: &eframe::CreationContext) -> Self {
        Self::default()
    }

    fn startup(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut visuals: egui::Visuals = egui::Visuals::dark();
        // visuals.panel_fill = Color32::from_rgba_unmultiplied(24, 36, 41, 255);
        ctx.set_visuals(visuals);
    }

    fn setup_ai(&mut self, nn_structure: GraphStructure) {
        if (nn_structure.validate()) {
            self.ai = Some(NeuralNetwork::new(nn_structure));
        }
        self.training_session.set_nn(self.ai.as_ref().unwrap());
        self.window_data.training_session_num_epochs = self.training_session.get_num_epochs();
        self.window_data.training_session_batch_size = self.training_session.get_batch_size();
        self.window_data.training_session_learn_rate = self.training_session.get_learn_rate();
    }

    fn draw_ui(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) -> InnerResponse<()> {
        let response = egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(&mut self.window_data.show_ai, "Show AI");
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
        if self.window_data.show_traning_dataset {
            self.window_training_set.draw_ui(
                ctx,
                &mut self.training_dataset,
                &mut self.window_data.training_dataset_split_thresholds_0,
                &mut self.window_data.training_dataset_split_thresholds_1,
            );
        }

        let vec: Vec<_> = self
            .training_dataset
            .get_datapoint_iter()
            .map(|f| f.clone())
            .collect();
        self.training_session.set_training_data(&vec[..]);
        if self.window_data.show_training_session {
            self.window_training_session.draw_ui(
                ctx,
                &mut self.training_session,
                &mut self.state,
                &mut self.training_thread,
            );
        }

        if self.window_data.show_ai {
            self.window_ai
                .draw_ui(ctx, self.ai.as_mut(), &self.training_dataset);
        }
        if self.window_data.show_training_graph {
            self.window_training_graph.draw_ui(ctx);
        }

        response
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
