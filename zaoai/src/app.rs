// hide console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use crate::{
    app_windows::{WindowAiSetupPresets, WindowAiSetupPresetsCtx},
    zneural_network::{
        activation::ActivationFunctionType,
        cost::CostFunction,
        datapoint::{TrainingData, TrainingDataset},
        is_correct::ConfusionEvaluator,
        layer::{BiasInit, WeightInit},
        neuralnetwork::load_neural_network,
    },
};
use eframe::{
    egui::{self, InnerResponse, Slider},
    epaint::Rect,
};
use zaoai_types::ai_labels::LayerTypeCPU;

use std::{
    ops::RangeInclusive,
    str::FromStr,
    sync::{Arc, Mutex},
};

use crate::{
    app_windows::{
        DrawableWindow, WindowAi, WindowAiCtx, WindowTrainingGraph, WindowTrainingGraphCtx,
        WindowTrainingSession, WindowTrainingSessionCtx, WindowTrainingSet, WindowTrainingSetCtx,
    },
    egui_ext::add_slider_sized,
    zneural_network::{
        datapoint::DataPoint,
        neuralnetwork::{GraphStructure, NeuralNetwork},
        thread::{TrainingThreadController, TrainingThreadPayload},
        training::{TrainingSession, TrainingState},
    },
};

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MenuWindowData {
    // Main Menu
    pub graph_structure_string: String,
    pub show_ai: bool,
    // Training Graph
    pub show_training_graph: bool,
    // Training Session
    pub show_training_session: bool,
    pub training_session_num_epochs: usize,
    pub training_session_batch_size: usize,
    pub training_session_learn_rate: LayerTypeCPU,
    // Training Dataset
    pub show_traning_dataset: bool,
    pub training_dataset_split_thresholds_0: f64,
    pub training_dataset_split_thresholds_1: f64,
    // AI options
    pub ai_use_softmax_output: bool,
    pub ai_activation_function: ActivationFunctionType,
    pub ai_cost_fn: CostFunction,
    pub ai_dropout_prob: LayerTypeCPU,
    pub ai_is_correct_fn: ConfusionEvaluator,
    pub ai_weight_init: WeightInit,
    pub ai_bias_init: BiasInit,
    // Setup Presets
    pub show_setup_presets: bool,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    window_setup_presets: WindowAiSetupPresets,

    payload_test_buffer: Vec<TrainingThreadPayload>,
}

impl eframe::App for ZaoaiApp {
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(60 * 3)
    }

    #[cfg(not(feature = "linux-profile"))]
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        use crate::zneural_network::neuralnetwork::save_neural_network;

        const NUM_SAVING: usize = 3;
        log::info!("[0/{NUM_SAVING}] Save Initiated");

        if let Some(nn) = &self.ai {
            const DEFAULT_NN_FILEPATH: &'static str = "NN/save.znn";
            let save_nn_filepath = DEFAULT_NN_FILEPATH;
            log::info!("[1/{NUM_SAVING}] Saving neural network: {save_nn_filepath}");
            if let Err(e) = save_neural_network(nn, save_nn_filepath) {
                log::error!("Failed to save neural network to {save_nn_filepath}: {e}");
            }
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
                let (_response, _rect) = self.draw_ui(ctx, frame);
                // ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(rect.size()));
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
                                self.training_session.nn = Some(ai.clone());
                                self.training_session.is_correct_fn =
                                    self.window_data.ai_is_correct_fn;
                                if (
                                    ai.graph_structure.input_nodes,
                                    ai.graph_structure.output_nodes,
                                ) == training_dataset_dim
                                {
                                    // Copy the session for TrainingThread to take care of
                                    if self.training_thread.begin_training(&self.training_session) {
                                        self.training_session.set_state(TrainingState::Training);
                                    } else {
                                        self.training_session.set_state(TrainingState::Idle);
                                    }
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
                        let mut received_any_training = false;
                        let mut train_disconnected = true;
                        let mut received_any_validation = false;
                        let mut train_disconnected = true;
                        if let Some(rx_training) = self.training_thread.rx_training_payload.as_mut()
                        {
                            (received_any_training, train_disconnected) = process_payload_channel(
                                rx_training,
                                &mut self.training_thread.payload_training_buffer,
                                "Training channel disconnected",
                            );
                            if train_disconnected {
                                self.training_thread.rx_training_payload = None;
                            }
                        } else {
                            log::error!("TrainingState::Training but could not get Training Payload Reciever");
                        }
                        if let Some(rx_validation) =
                            self.training_thread.rx_validation_payload.as_mut()
                        {
                            let (received_any_validation, validation_disconnected) =
                                process_payload_channel(
                                    rx_validation,
                                    &mut self.training_thread.payload_validation_buffer,
                                    "Validation channel disconnected",
                                );
                            if validation_disconnected {
                                self.training_thread.rx_validation_payload = None;
                            }
                        } else {
                            log::error!("TrainingState::Training but could not get Validation Payload Reciever");
                        }

                        // if received_any_training || received_any_validation {
                        // Need to repaint each frame for now. Otherwise crash due to channel disconnect on complete
                        ctx.request_repaint();
                        // }

                        let training_in_progress = self.training_thread.training_in_progress();
                        if !training_in_progress && received_any_training {
                            let training_payload_buffer =
                                &self.training_thread.payload_training_buffer;
                            if training_payload_buffer.len() != training_payload_buffer.capacity() {
                                log::error!("payload_buffer.len() != payload_buffer.capacity(), some data was not put in payload_buffer");
                            }
                            self.training_session.set_state(TrainingState::Finish);
                        }
                    }
                    TrainingState::Finish => {
                        log::trace!("Training Finished! Waiting for result");

                        if let Some(rx_neural_network) =
                            self.training_thread.rx_neuralnetwork.as_ref()
                        {
                            let result = rx_neural_network.try_recv();
                            match result {
                                Ok(result) => {
                                    log::trace!("Training result recieved, updating AI");
                                    self.ai = Some(result);

                                    self.training_session.set_state(TrainingState::Idle);
                                    self.state = AppState::Idle;
                                }
                                Err(e) => match e {
                                    std::sync::mpsc::TryRecvError::Empty => { /*Waiting for sender*/
                                    }
                                    std::sync::mpsc::TryRecvError::Disconnected => log::error!(
                                        "Failed to recieve training finish data data: {}",
                                        e
                                    ),
                                },
                            }
                        } else {
                            log::error!("Failed to get training thread neural network reciever");
                        }

                        ctx.request_repaint();
                    }
                }

                let (_response, _rect) = self.draw_ui(ctx, frame);
                // ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(rect.size()));
            }
            AppState::Exit => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
            _default => {
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
                ai_use_softmax_output: false,
                ai_activation_function: ActivationFunctionType::ReLU,
                ai_dropout_prob: 0.5,
                ai_is_correct_fn: ConfusionEvaluator::LargestLabel,
                ai_cost_fn: CostFunction::Mse,
                ai_weight_init: WeightInit::default(),
                ai_bias_init: BiasInit::default(),
                show_setup_presets: true,
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
                ConfusionEvaluator::LargestLabel,
                0,
            ),
            window_training_graph: WindowTrainingGraph::default(),
            window_ai: WindowAi {
                test_nn_rx: None,
                test_nn_handle: None,
                test_nn_done: Arc::new(Mutex::new(None)),
                test_nn_abort_tx: None,
                test_nn_graph: true,
            },
            window_training_set: WindowTrainingSet::default(),
            window_training_session: WindowTrainingSession {},
            last_ai_filepath: None,
            training_thread: TrainingThreadController::default(),
            window_setup_presets: WindowAiSetupPresets::default(),
            payload_test_buffer: vec![],
        }
    }
}

fn process_payload_channel<T>(
    rx: &std::sync::mpsc::Receiver<T>,
    buffer: &mut Vec<T>,
    warn_msg: &str,
) -> (bool, bool) {
    const MAX_TO_PROCESS: usize = 1000;
    let mut received = false;
    let mut disconnected = false;

    for _ in 0..MAX_TO_PROCESS {
        match rx.try_recv() {
            Ok(item) => {
                buffer.push(item);
                received = true;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => break,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                log::warn!("{}", warn_msg);
                disconnected = true;
                break;
            }
        }
    }
    (received, disconnected)
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        Self::default()
    }

    // Should only be called once per application launch
    fn startup(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Try load NN from disk
        if let Some(last_ai_filepath) = &self.last_ai_filepath {
            log::info!("Loading neural network from: {last_ai_filepath}...");
            match load_neural_network(&last_ai_filepath) {
                Ok(_) => log::info!("Loaded neural network from: {last_ai_filepath}"),
                Err(e) => log::error!("{e}"),
            }
        }

        let visuals: egui::Visuals = egui::Visuals::dark();
        // visuals.panel_fill = Color32::from_rgba_unmultiplied(24, 36, 41, 255);
        ctx.set_visuals(visuals);
    }

    fn setup_ai(&mut self, nn_structure: GraphStructure) {
        log::info!("setup_ai");
        self.ai = Some(NeuralNetwork::new(
            nn_structure,
            self.window_data.ai_activation_function,
            self.window_data.ai_cost_fn,
            Some(self.window_data.ai_dropout_prob),
            self.window_data.ai_weight_init,
            self.window_data.ai_bias_init,
        ));
        self.training_session.set_nn(self.ai.as_ref().unwrap());
        self.window_data.training_session_num_epochs = self.training_session.get_num_epochs();
        self.window_data.training_session_batch_size = self.training_session.get_batch_size();
        self.window_data.training_session_learn_rate = self.training_session.get_learn_rate();
    }

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        _frame: &mut eframe::Frame,
    ) -> (InnerResponse<InnerResponse<()>>, Rect) {
        let mut min_rect = Rect::ZERO;
        let response = egui::CentralPanel::default().show(ctx, |ui| {
            let mut change_state = false;

            ui.vertical(|ui| {
                ui.checkbox(&mut self.window_data.show_ai, "Show AI");
                ui.checkbox(&mut self.window_data.show_traning_dataset, "Show Dataset");
                ui.checkbox(&mut self.window_data.show_training_session, "Show Training");
                ui.checkbox(
                    &mut self.window_data.show_training_graph,
                    "Show Training Graph",
                );

                let name_label = ui.label("Create new NN with layers");
                change_state |= ui
                    .text_edit_singleline(&mut self.window_data.graph_structure_string)
                    .labelled_by(name_label.id)
                    .lost_focus();

                ui.horizontal(|ui| {
                    let dropout_slider = add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(&mut self.window_data.ai_dropout_prob, 0.01..=0.5)
                            .clamping(egui::SliderClamping::Never)
                            .min_decimals(2)
                            .max_decimals_opt(Some(5)),
                    );
                    change_state |= dropout_slider.drag_stopped();
                    ui.label("Dropout %");
                });

                change_state |= ui
                    .checkbox(
                        &mut self.window_data.ai_use_softmax_output,
                        "Use softmax output",
                    )
                    .changed();

                macro_rules! combo {
                    ($label:expr, $field:expr, $variants:expr) => {{
                        let before = $field;
                        egui::ComboBox::from_label($label)
                            .selected_text($field.to_string())
                            .show_ui(ui, |ui| {
                                for variant in $variants {
                                    ui.selectable_value(&mut $field, variant, variant.to_string());
                                }
                            });
                        before != $field
                    }};
                }

                change_state |= combo!(
                    "Activation Function",
                    self.window_data.ai_activation_function,
                    [
                        ActivationFunctionType::ReLU,
                        ActivationFunctionType::Sigmoid
                    ]
                );
                change_state |= combo!(
                    "Is Correct Fn",
                    self.window_data.ai_is_correct_fn,
                    [
                        ConfusionEvaluator::LargestLabel,
                        ConfusionEvaluator::Zlbl,
                        ConfusionEvaluator::ZlblLoose
                    ]
                );
                change_state |= combo!(
                    "Cost Fn",
                    self.window_data.ai_cost_fn,
                    [
                        CostFunction::Mse,
                        CostFunction::CrossEntropyMulticlass,
                        CostFunction::CrossEntropyBinary
                    ]
                );
                change_state |= combo!(
                    "Weight Init",
                    self.window_data.ai_weight_init,
                    WeightInit::all().into_iter().map(|v| *v)
                );
                change_state |= combo!(
                    "Bias Init",
                    self.window_data.ai_bias_init,
                    BiasInit::all().into_iter().map(|v| *v)
                );

                if change_state {
                    self.state = AppState::SetupAi;
                }
            })
        });

        min_rect = min_rect.union(response.inner.response.rect);

        macro_rules! draw_win {
            ($show:expr, $window:expr, $ctx_struct:expr) => {
                if $show {
                    $window.with_ctx(ctx, $ctx_struct, |this, state_ctx| {
                        if let Some(r) = this.draw_ui(ctx, state_ctx) {
                            min_rect = min_rect.union(r.response.rect);
                        }
                    });
                }
            };
        }

        draw_win!(
            self.window_data.show_traning_dataset,
            self.window_training_set,
            &mut WindowTrainingSetCtx {
                training_data: &mut self.training_data
            }
        );

        draw_win!(
            self.window_data.show_training_session,
            self.window_training_session,
            &mut WindowTrainingSessionCtx {
                training_session: &mut self.training_session,
                app_state: &mut self.state,
                training_thread: &mut self.training_thread,
            }
        );

        draw_win!(
            self.window_data.show_ai,
            self.window_ai,
            &mut WindowAiCtx {
                ai: &mut self.ai,
                test_button_training_data: &Some(&self.training_data),
                ai_is_corret_fn: &self.window_data.ai_is_correct_fn,
                payload_test_buffer: &mut self.payload_test_buffer,
            }
        );

        if self.window_data.show_ai {
            if let Some(rx) = &self.window_ai.test_nn_rx {
                self.payload_test_buffer.extend(
                    rx.try_iter()
                        .take(1000)
                        .inspect(|_| log::trace!("Test data received!")),
                );
            }
        }

        if self.window_ai.test_nn_handle.is_some() {
            if let Ok(mut nn_done) = self.window_ai.test_nn_done.try_lock() {
                if let Some(r) = nn_done.take() {
                    log::info!("test_nn results recieved!\n{r}");
                    if let Some(ai) = &mut self.ai {
                        ai.last_test_results = Some(r);
                    }
                    self.window_ai.test_nn_handle = None;
                }
            }
        }

        draw_win!(
            self.window_data.show_training_graph,
            self.window_training_graph,
            &mut WindowTrainingGraphCtx {
                payload_training_buffer: &mut self.training_thread.payload_training_buffer,
                payload_validation_buffer: &mut self.training_thread.payload_validation_buffer,
                payload_test_buffer: &mut self.payload_test_buffer,
            }
        );

        draw_win!(
            self.window_data.show_setup_presets,
            self.window_setup_presets,
            &mut WindowAiSetupPresetsCtx {
                window_data: &mut self.window_data,
                state: &mut self.state,
            }
        );

        (response, min_rect)
    }
}
