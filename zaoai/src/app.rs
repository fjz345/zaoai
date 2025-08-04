// hide console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use crate::{
    error::Result,
    zneural_network::{
        datapoint::{TrainingData, TrainingDataset},
        is_correct::IsCorrectFn,
        layer::ActivationFunctionType,
        neuralnetwork::load_neural_network,
        training::FloatDecay,
    },
};
use eframe::{
    egui::{self, style::Widgets, InnerResponse, RawInput, Response, Slider},
    epaint::{Color32, Pos2, Rect},
    glow::TESS_EVALUATION_TEXTURE,
    App,
};
use egui_plot::PlotPoint;
use graphviz_rust::{dot_structures::Graph, print};
use ndarray::{Array2, ArrayBase, Dim, OwnedRepr};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::{
    fs::File,
    io::{self, Read, Write},
    ops::RangeInclusive,
    str::FromStr,
    sync::mpsc::Receiver,
    thread::JoinHandle,
    time::Duration,
};
use zaoai_types::ai_labels::ZaoaiLabel;

use crate::{
    app_windows::{
        DrawableWindow, WindowAi, WindowAiCtx, WindowTrainingGraph, WindowTrainingGraphCtx,
        WindowTrainingSession, WindowTrainingSessionCtx, WindowTrainingSet, WindowTrainingSetCtx,
    },
    egui_ext::{add_slider_sized, Interval},
    mnist::get_mnist,
    zneural_network::{
        datapoint::{create_2x2_test_datapoints, split_datapoints, DataPoint},
        neuralnetwork::{GraphStructure, NeuralNetwork},
        thread::{TrainingThreadController, TrainingThreadPayload},
        training::{self, TrainingSession, TrainingState},
    },
};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct MenuWindowData {
    // Main Menu
    graph_structure_string: String,
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
    // AI options
    ai_use_softmax: bool,
    ai_activation_function: ActivationFunctionType,
    ai_dropout_proc: f32,
    ai_is_correct_fn: IsCorrectFn,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ZaoaiApp {
    #[cfg_attr(feature = "serde", serde(skip))]
    state: AppState,
    #[cfg_attr(feature = "serde", serde(skip))]
    ai: Option<NeuralNetwork>,
    last_ai_filepath: Option<String>,
    window_data: MenuWindowData,
    #[cfg_attr(feature = "serde", serde(skip))]
    training_data: TrainingData,
    training_session: TrainingSession,
    #[cfg_attr(feature = "serde", serde(skip))]
    training_thread: TrainingThreadController,
    window_training_graph: WindowTrainingGraph,
    window_ai: WindowAi,
    window_training_set: WindowTrainingSet,
    window_training_session: WindowTrainingSession,
}

impl eframe::App for ZaoaiApp {
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(60 * 3)
    }

    #[cfg(not(feature = "linux-profile"))]
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        use crate::zneural_network::neuralnetwork::save_neural_network;
        use std::path::Path;

        const NUM_SAVING: usize = 3;
        log::info!("[0/{NUM_SAVING}] Save Initiated");

        if let Some(nn) = &self.ai {
            const DEFAULT_NN_FILEPATH: &'static str = "NN/save.znn";
            let save_nn_filepath = DEFAULT_NN_FILEPATH;
            log::info!("[1/{NUM_SAVING}] Saving neural network: {save_nn_filepath}");
            save_neural_network(nn, save_nn_filepath);
            self.last_ai_filepath = Some(save_nn_filepath.to_owned());
        } else {
            log::info!("[1/{NUM_SAVING}] Neural network not saved, not set");
        }

        if cfg!(feature = "serde") {
            #[cfg(feature = "serde")]
            {
                let json_result = serde_json::to_string(self);
                match json_result {
                    Ok(json) => {
                        log::info!("[2/{NUM_SAVING}] Saving ZaoaiApp Json to persistant storage");
                        storage.set_string(eframe::APP_KEY, json);
                    }
                    Err(e) => {
                        log::debug!("[2/{NUM_SAVING}] Persistant storage failed");
                        log::debug!("{e}");
                    }
                }
            }
        } else {
            log::info!("[2/{NUM_SAVING}] Persistant storage not saved (not enabled)");
        }

        log::info!("[{NUM_SAVING}/{NUM_SAVING}] Save Complete!");
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match self.state {
            AppState::Idle | AppState::Training => {}
            AppState::Startup | AppState::SetupAi | AppState::Testing | AppState::Exit => {
                log::info!("AppState::{}", &self.state)
            }
        }

        match self.state {
            AppState::Startup => {
                self.startup(ctx, frame);
                self.state = AppState::SetupAi;
            }
            AppState::SetupAi => {
                // Todo: make a function to format nn_structure
                {
                    let mut formatted_nn_structure = self
                        .window_data
                        .graph_structure_string
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

                    if formatted_nn_structure.len() >= 2 {
                        let graph = GraphStructure::new(&formatted_nn_structure);
                        if graph.validate() {
                            self.setup_ai(graph);
                        } else {
                            log::info!("Graph not valid, setup skipped");
                        }
                    } else {
                        log::error!("AI might not be initialized correctly, formatted_nn_structure.len() < 2");
                    }
                }

                self.state = AppState::Idle;
            }
            AppState::Idle => {
                let (response, rect) = self.draw_ui(ctx, frame);

                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(rect.size()));
            }
            AppState::Training => {
                match self.training_session.get_state() {
                    TrainingState::Idle => {
                        log::trace!("TrainingState::Idle");
                    }

                    TrainingState::StartTraining => {
                        if let Some(ai) = &self.ai {
                            if !self.training_thread.training_in_progress() {
                                let training_dataset_dim =
                                    self.training_data.get_in_out_dimensions();
                                self.training_session
                                    .set_training_data(self.training_data.clone());
                                self.training_session.is_correct_fn =
                                    self.window_data.ai_is_correct_fn;
                                if (
                                    ai.graph_structure.input_nodes,
                                    ai.graph_structure.output_nodes,
                                ) == training_dataset_dim
                                {
                                    // Copy the session for TrainingThread to take care of
                                    self.training_thread.begin_training(&self.training_session);
                                    self.training_session.set_state(TrainingState::Training);
                                } else {
                                    log::error!("Cannot start training, dimension missmatch (NN: {}/{}) != (DP: {}/{})", ai.graph_structure.input_nodes, ai.graph_structure.output_nodes, training_dataset_dim.0, training_dataset_dim.1);
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
                            .rx_payload
                            .as_mut()
                            .expect("ERROR")
                            .try_recv();

                        let in_progress = self.training_thread.training_in_progress();
                        if let Ok(result_metadata) = result_metadata {
                            let payload_buffer = &mut self.training_thread.payload_buffer;
                            payload_buffer.push(result_metadata);

                            if !in_progress {
                                assert_eq!(payload_buffer.len(), payload_buffer.capacity());
                                self.training_session.set_state(TrainingState::Finish);
                            }
                        }
                    }
                    TrainingState::Finish => {
                        log::info!("Training Finished");

                        let result = self
                            .training_thread
                            .rx_neuralnetwork
                            .as_ref()
                            .expect("ERROR")
                            .try_recv();
                        if result.is_ok() {
                            self.ai = Some(result.unwrap());
                        } else {
                            panic!("Unexpected error");
                        }

                        self.training_session.set_state(TrainingState::Idle);
                        self.state = AppState::Idle;
                    }
                    TrainingState::Abort => {
                        panic!("Not Implemented");
                    }
                }

                let (response, rect) = self.draw_ui(ctx, frame);
                ctx.request_repaint();
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(rect.size()));
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
        let graph_structure = GraphStructure::new(&[2, 3, 2]);
        Self {
            state: AppState::Startup,
            ai: None,
            window_data: MenuWindowData {
                graph_structure_string: graph_structure.to_string(),
                show_training_graph: true,
                show_training_session: true,
                training_session_num_epochs: 2,
                training_session_batch_size: 1000,
                training_session_learn_rate: 0.2,
                show_traning_dataset: true,
                training_dataset_split_thresholds_0: 0.75,
                training_dataset_split_thresholds_1: 0.9,
                show_ai: true,
                ai_use_softmax: false,
                ai_activation_function: ActivationFunctionType::ReLU,
                ai_dropout_proc: 0.0,
                ai_is_correct_fn: IsCorrectFn::MaxVal,
            },
            training_data: TrainingData::Physical(TrainingDataset::new(
                &[DataPoint {
                    inputs: vec![0.0; 2],
                    expected_outputs: vec![0.0; 2],
                }; 0],
            )),

            training_session: TrainingSession::new(
                None,
                TrainingData::default(),
                100,
                1000,
                0.2,
                None,
                0.0,
            ),
            window_training_graph: WindowTrainingGraph::default(),
            window_ai: WindowAi {},
            window_training_set: WindowTrainingSet::default(),
            window_training_session: WindowTrainingSession {},
            last_ai_filepath: None,
            training_thread: TrainingThreadController::default(),
        }
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default, Debug, Clone, Copy, PartialEq, strum_macros::Display)]
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

    // Should only be called once per application launch
    fn startup(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Try load NN from disk
        if let Some(last_ai_filepath) = &self.last_ai_filepath {
            log::info!("Loading neural network from: {last_ai_filepath}...");
            match load_neural_network(&last_ai_filepath) {
                Ok(r) => log::info!("Loaded neural network from: {last_ai_filepath}"),
                Err(e) => log::error!("{e}"),
            }
        }

        let mut visuals: egui::Visuals = egui::Visuals::dark();
        // visuals.panel_fill = Color32::from_rgba_unmultiplied(24, 36, 41, 255);
        ctx.set_visuals(visuals);
    }

    fn setup_ai(&mut self, nn_structure: GraphStructure) {
        log::info!("setup_ai");
        self.ai = Some(NeuralNetwork::new(
            nn_structure,
            self.window_data.ai_activation_function,
        ));
        self.training_session.set_nn(self.ai.as_ref().unwrap());
        self.window_data.training_session_num_epochs = self.training_session.get_num_epochs();
        self.window_data.training_session_batch_size = self.training_session.get_batch_size();
        self.window_data.training_session_learn_rate = self.training_session.get_learn_rate();
    }

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        frame: &mut eframe::Frame,
    ) -> (InnerResponse<InnerResponse<()>>, Rect) {
        let mut min_rect = Rect::ZERO;
        let response = egui::CentralPanel::default().show(ctx, |ui| {
            let mut change_state_to_setupai = false;

            ui.vertical(|ui| {
                ui.checkbox(&mut self.window_data.show_ai, "Show AI");
                ui.checkbox(&mut self.window_data.show_traning_dataset, "Show Dataset");

                ui.checkbox(&mut self.window_data.show_training_session, "Show Training");

                ui.checkbox(
                    &mut self.window_data.show_training_graph,
                    "Show Training Graph",
                );

                let name_label = ui.label("Create new NN with layers");
                let changed = ui
                    .text_edit_singleline(&mut self.window_data.graph_structure_string)
                    .labelled_by(name_label.id)
                    .lost_focus();
                change_state_to_setupai |= changed;

                ui.horizontal(|ui| {
                    let dropout_slider = add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut self.window_data.ai_dropout_proc,
                            RangeInclusive::new(0.01, 0.5),
                        )
                        .clamping(egui::SliderClamping::Never)
                        .min_decimals(2)
                        .max_decimals_opt(Some(5)),
                    );
                    let changed = dropout_slider.drag_stopped();

                    ui.label("Dropout %");

                    change_state_to_setupai |= changed;
                });

                let changed = ui
                    .checkbox(&mut self.window_data.ai_use_softmax, "Use softmax")
                    .changed();
                change_state_to_setupai |= changed;

                let act_before = self.window_data.ai_activation_function;
                let combo_response = egui::ComboBox::from_label("Activation Function")
                    .selected_text(self.window_data.ai_activation_function.to_string())
                    .show_ui(ui, |ui| {
                        for variant in [
                            ActivationFunctionType::ReLU,
                            ActivationFunctionType::Sigmoid,
                        ] {
                            ui.selectable_value(
                                &mut self.window_data.ai_activation_function,
                                variant,
                                variant.to_string(),
                            );
                        }
                    });

                let changed = act_before != self.window_data.ai_activation_function;
                change_state_to_setupai |= changed;

                if change_state_to_setupai {
                    self.state = AppState::SetupAi;
                }

                let is_correct_before = self.window_data.ai_is_correct_fn;
                let combo_response = egui::ComboBox::from_label("Is Correct Fn")
                    .selected_text(self.window_data.ai_is_correct_fn.to_string())
                    .show_ui(ui, |ui| {
                        for variant in [
                            IsCorrectFn::MaxVal,
                            IsCorrectFn::Zlbl,
                            IsCorrectFn::ZlblLoose,
                        ] {
                            ui.selectable_value(
                                &mut self.window_data.ai_is_correct_fn,
                                variant,
                                variant.to_string(),
                            );
                        }
                    });

                let changed = is_correct_before != self.window_data.ai_is_correct_fn;
            })
        });
        min_rect = min_rect.union(response.inner.response.rect);

        // Windows
        if self.window_data.show_traning_dataset {
            self.window_training_set.with_ctx(
                ctx,
                &mut WindowTrainingSetCtx {
                    training_data: &mut self.training_data,
                },
                |this, state_ctx| {
                    let response = this.draw_ui(ctx, state_ctx);
                    if let Some(r) = response {
                        min_rect = min_rect.union(r.response.rect);
                    }
                },
            );
        }

        if self.window_data.show_training_session {
            self.window_training_session.with_ctx(
                ctx,
                &mut WindowTrainingSessionCtx {
                    training_session: &mut self.training_session,
                    app_state: &mut self.state,
                    training_thread: &mut self.training_thread,
                },
                |this, state_ctx| {
                    let response = this.draw_ui(ctx, state_ctx);
                    if let Some(r) = response {
                        min_rect = min_rect.union(r.response.rect);
                    }
                },
            );
        }

        if self.window_data.show_ai {
            self.window_ai.with_ctx(
                ctx,
                &mut WindowAiCtx {
                    ai: &mut self.ai,
                    test_button_training_data: &Some(&self.training_data),
                    ai_is_corret_fn: &self.window_data.ai_is_correct_fn,
                },
                |this, state_ctx| {
                    let response = this.draw_ui(ctx, state_ctx);
                    if let Some(r) = response {
                        min_rect = min_rect.union(r.response.rect);
                    }
                },
            );
        }
        if self.window_data.show_training_graph {
            self.window_training_graph.with_ctx(
                ctx,
                &mut WindowTrainingGraphCtx {
                    training_thread: &self.training_thread,
                },
                |this, state_ctx| {
                    let response = this.draw_ui(ctx, state_ctx);
                    if let Some(r) = response {
                        min_rect = min_rect.union(r.response.rect);
                    }
                },
            );
        }
        (response, min_rect)
    }
}
