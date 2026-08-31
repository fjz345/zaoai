use std::{ ops::RangeInclusive, path::PathBuf, sync::{Arc, Mutex, mpsc::{self, Receiver, Sender}}, thread::JoinHandle};

use crate::{
    app::{AppState, MenuWindowData}, egui_ext::{Interval, add_slider_sized}, mnist::get_mnist, zneural_network::{
        activation::ActivationFunctionType, cost::CostFunction, datapoint::{
            DataPoint, TrainingData, TrainingDataset, VirtualTrainingDataset, create_2x2_test_datapoints,
        }, is_correct::ConfusionEvaluator, layer::{ BiasInit, WeightInit}, neuralnetwork::{GraphStructure, NeuralNetwork, NeuralNetworkPingPong}, thread::{TrainingThreadController, TrainingThreadPayload}, training::{FloatDecay, TestResults, TrainingSession, TrainingState, test_nn}
    },
};
use zaoai_types::ai_labels::LayerTypeCPU;
use eframe::egui::{self, Align, Button, Color32, InnerResponse, Layout, Sense, Slider, SliderClamping};
use egui_plot::{Corner, Legend, PlotResponse};
use egui_plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints};

use zaoai_types::{
    ai_labels::{AnimeDataPoint, ZaoaiLabelsLoader},
    sound::get_spectrogram_dims,
    spectrogram::{generate_spectrogram, SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, S_SPECTROGRAM_NUM_BINS},
};

pub trait DrawableWindow<'a> {
    type Ctx;

    fn with_ctx<F>(&mut self, _egui_ctx: &egui::Context, ctx: &mut Self::Ctx, f: F)
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
    pub(crate) payload_training_buffer: &'a mut Vec<TrainingThreadPayload>,
    pub(crate) payload_validation_buffer: &'a mut Vec<TrainingThreadPayload>,
    pub(crate) payload_test_buffer: &'a mut Vec<TrainingThreadPayload>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
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

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default)]
pub struct WindowTrainingGraph {
    // Training
    cached_plot_points_accuracy: Vec<SerdePlotPoint>, // cached since want to restore the graphs on app load
    cached_plot_points_cost: Vec<SerdePlotPoint>,
    cached_plot_points_last_loss: Vec<SerdePlotPoint>,
    cached_plot_points_learn_rate: Vec<SerdePlotPoint>,
    cached_plot_points_f1_score: Vec<SerdePlotPoint>,
    
    // Validation

    // Test
}


macro_rules! gen_line {
    (
        self = $self:ident,
        payload = $payload:expr,
        cache_field = $cache_field:ident,
        generator = $gen_fn:path,
        label = $label:expr,
        color = $color:expr,
    ) => {{
        use crate::app_windows::PlotPoints::Owned;

        $self.$cache_field = $gen_fn($payload)
            .into_iter()
            .map(Into::into)
            .collect();

        let plot_points = Owned(
            $self.$cache_field
                .clone()
                .into_iter()
                .map(Into::into)
                .collect(),
        );

        Line::new($label, plot_points)
            .color($color)
            .id($label)
    }};
}

impl WindowTrainingGraph
{
    fn generate_common_lines(&mut self, payload_buffer: &Vec<TrainingThreadPayload>) -> Vec<Line<'_>>
    {
        let line_accuracy = gen_line! {
            self = self,
            payload = &payload_buffer,
            cache_field = cached_plot_points_accuracy,
            generator = generate_accuracy_plotpoints_from_training_thread_payloads,
            label = "Accuracy %",
            color = Color32::LIGHT_GREEN,
        };
        let line_cost = gen_line! {
            self = self,
            payload = &payload_buffer,
            cache_field = cached_plot_points_cost,
            generator = generate_cost_plotpoints_from_training_thread_payloads,
            label = "Cost",
            color = Color32::LIGHT_RED,
        };
        let line_last_loss = gen_line! {
            self = self,
            payload = &payload_buffer,
            cache_field = cached_plot_points_last_loss,
            generator = generate_last_loss_plotpoints_from_training_thread_payloads,
            label = "Last Loss",
            color = Color32::LIGHT_YELLOW,
        };
        let line_f1_score = gen_line! {
            self = self,
            payload = &payload_buffer,
            cache_field = cached_plot_points_f1_score,
            generator = generate_f1_score_plotpoints_from_training_thread_payloads,
            label = "F1 Score",
            color = Color32::LIGHT_BLUE,
        };

        vec![line_accuracy, line_cost, line_last_loss, line_f1_score]
    }

    fn render_plot(
        ui: &mut egui::Ui,
        title: &str,
        x_label: &str,
        payload_buffer: &mut Vec<TrainingThreadPayload>,
        common_lines: Vec<Line<'_>>,
        extra_lines: impl IntoIterator<Item = Line<'static>>,
    ) -> PlotResponse<()> {
        let toggle_id = egui::Id::new(format!("{}_full_view_toggle", title));
        let mut full_view_toggle_value = ui.memory_mut(|m| {
            m.data.get_persisted(toggle_id).unwrap_or(false)
        });

        ui.horizontal(|ui| {
            ui.label(title);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Clear").clicked() {
                    payload_buffer.clear();
                }
                if ui.toggle_value(&mut full_view_toggle_value, "Full View").clicked() {
                    ui.memory_mut(|m| {
                        m.data.insert_persisted(toggle_id, full_view_toggle_value)
                    });
                }
            });
        });

        Self::create_plot_training(title)
            .legend(Legend::default().position(Corner::LeftBottom).follow_insertion_order(true))
            .x_axis_label(x_label)
            .include_x(0.0)
            .show(ui, |plot_ui| {
            if !full_view_toggle_value {
                // Egui for some reason does not support reading legend toggle state
                // Would like to cull the metrics that are hidden
                let payload_max = payload_buffer
                    .last()
                    .map(|p| p.payload_index as f64)
                    .unwrap_or(0.0);

                const COUNT: f64 = 10.0;
                let min_x = (payload_max - COUNT).max(0.0);

                plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                    [min_x, f64::NEG_INFINITY],
                    [payload_max, f64::INFINITY],
                ));
                plot_ui.set_auto_bounds([false, true]); 
            } else {
                plot_ui.set_auto_bounds([true, true]);
            }

            for line in common_lines {
                plot_ui.line(line);
            }
            for line in extra_lines {
                plot_ui.line(line);
            }
        })
    }

    fn show_training_plot(
        &mut self,
        ui: &mut egui::Ui,
        state_ctx: &mut WindowTrainingGraphCtx,
    ) -> PlotResponse<()> {
        let buffer = &mut *state_ctx.payload_training_buffer;
        let learn_rate_line = gen_line! {
                self = self,
                payload = &buffer,
                cache_field = cached_plot_points_learn_rate,
                generator = generate_learn_rate_plotpoints_from_training_thread_payloads,
                label = "Learn Rate",
                color = Color32::LIGHT_GRAY,
            };
        let common_lines = self.generate_common_lines(buffer);

        Self::render_plot(ui, "Training", "Epoch", buffer, common_lines, vec![learn_rate_line])
    }

    fn show_validation_plot(
        &mut self,
        ui: &mut egui::Ui,
        _ctx: &egui::Context,
        state_ctx: &mut WindowTrainingGraphCtx,
    ) -> PlotResponse<()> {
        let buffer = &mut *state_ctx.payload_validation_buffer;
        let common_lines = self.generate_common_lines(buffer);

        Self::render_plot(ui, "Validation", "Epoch", buffer, common_lines, vec![])
    }

    fn show_test_plot(
        &mut self,
        ui: &mut egui::Ui,
        _ctx: &egui::Context,
        state_ctx: &mut WindowTrainingGraphCtx,
    ) -> PlotResponse<()> {
        let buffer = &mut *state_ctx.payload_test_buffer;
        let common_lines = self.generate_common_lines(buffer);

        Self::render_plot(ui, "Testing", "Datapoint", buffer, common_lines, vec![])
    }
}

impl<'a> DrawableWindow<'a> for WindowTrainingGraph {
    type Ctx = WindowTrainingGraphCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        let window = egui::Window::new("Training Graph").default_pos(egui::Pos2::new(1000.0, 0.0)).show(ctx, |ui| {
            let _training_plot = self.show_training_plot(ui, state_ctx);
            let _validation_plot = self.show_validation_plot(ui, ctx, state_ctx);
            let _test_plot = self.show_test_plot(ui, ctx, state_ctx);
        });

        window
    }
}

impl WindowTrainingGraph {
    fn create_plot_training<'a>(id_source: impl std::hash::Hash) -> Plot<'a> {
        const INCLUDE_Y_PADDING: f64 = 0.06;
        Plot::new(id_source)
            .allow_drag(false)
            .allow_zoom(false)
            .allow_scroll(false)
            .allow_boxed_zoom(false)
            .allow_double_click_reset(false)
            .center_x_axis(false)
            .include_y(0.0 - INCLUDE_Y_PADDING)
            .include_y(1.0 + INCLUDE_Y_PADDING)
            .default_y_bounds(0.0 - INCLUDE_Y_PADDING, 1.0 + INCLUDE_Y_PADDING)
            .auto_bounds([true, true])
            .include_x(0.0)
            .y_grid_spacer(
                Self::create_plot_training_y_spacer_func as fn(GridInput) -> Vec<GridMark>,
            )
            .width(500.0)
            .height(300.0)
    }

    fn create_plot_training_y_spacer_func(grid: GridInput) -> Vec<GridMark> {
        let (min, max) = grid.bounds;
        let span = max - min;

        if span <= 0.0 || !span.is_finite() {
            return Vec::new();
        }

        let raw_step = span / 8.0;
        let exponent = raw_step.log10().floor();
        let scale = 10.0_f64.powf(exponent);

        let normalized = raw_step / scale;
        let major_step = if normalized < 1.5 {
            scale
        } else if normalized < 3.5 {
            2.0 * scale
        } else if normalized < 7.5 {
            5.0 * scale
        } else {
            10.0 * scale
        };

        let minor_step = major_step / 5.0;
        let mut marks = Vec::new();

        let start_minor = (min / minor_step).floor() as i64;
        let end_minor = (max / minor_step).ceil() as i64;

        for i in start_minor..=end_minor {
            let val = i as f64 * minor_step;
            if val >= min && val <= max {
                marks.push(GridMark {
                    value: val,
                    step_size: minor_step,
                });
            }
        }

        let start_major = (min / major_step).floor() as i64;
        let end_major = (max / major_step).ceil() as i64;

        for i in start_major..=end_major {
            let val = i as f64 * major_step;
            if val >= min && val <= max {
                marks.push(GridMark {
                    value: val,
                    step_size: major_step,
                });
            }
        }

        marks
    }
}

pub struct WindowAiCtx<'a> {
    pub ai: &'a mut Option<NeuralNetwork>,
    pub test_button_training_data: &'a Option<&'a TrainingData>,
    pub ai_is_corret_fn: &'a ConfusionEvaluator,
    pub payload_test_buffer: &'a mut Vec<TrainingThreadPayload>,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowAi {
    #[cfg_attr(feature = "serde", serde(skip))]
    pub test_nn_rx: Option<Receiver<TrainingThreadPayload>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub test_nn_abort_tx: Option<Sender<()>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub test_nn_handle: Option<JoinHandle<()>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub test_nn_done: Arc<Mutex<Option<TestResults>>>,

    pub test_nn_graph: bool,
}

impl<'a> DrawableWindow<'a> for WindowAi {
    type Ctx = WindowAiCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        egui::Window::new("ZaoAI").default_pos(egui::pos2(700.0, 0.0)).show(ctx, |ui| {

            ui.with_layout(Layout::right_to_left(Align::Min), |ui| {
                let delete_button = Button::new("Delete").sense(Sense::click());
                if ui.add(delete_button).clicked() {
                    *state_ctx.ai = None;
                }
            });

            ui.label(state_ctx.ai.as_ref().and_then(|f|Some(f.to_string())).unwrap_or_default());
            if let Some(ai) = &mut state_ctx.ai {
                enum TestButtonState {
                    TestingDone,
                    AbortTest,
                    StartTest,
                }
                impl TestButtonState {
                    fn from_handles(test_nn_handle: &Option<std::thread::JoinHandle<()>>, test_done_val: &Option<TestResults>) -> Self {
                        match (test_nn_handle, test_done_val) {
                            (Some(_), Some(_)) |
                            (Some(_), None) => TestButtonState::AbortTest,
                            (None, Some(_)) => TestButtonState::TestingDone,
                            (None, None) => TestButtonState::StartTest,
                        }
                    }

                    fn label(&self) -> &'static str {
                        match self {
                            TestButtonState::TestingDone => "Testing done!",
                            TestButtonState::AbortTest => "Abort Test",
                            TestButtonState::StartTest => "Start Test",
                        }
                    }
                }

                let test_done_val = {
                    let lock = self.test_nn_done.lock().unwrap();
                    lock.clone()
                };

                let mut ai_clone = ai.clone();
                ui.horizontal(|ui|{
                let button_state = TestButtonState::from_handles(&self.test_nn_handle, &test_done_val);
                let test_button = Button::new(button_state.label()).sense(Sense::click());
                let test_button_response = ui.add(test_button);
                let _test_graph_checkbox = ui.checkbox(&mut self.test_nn_graph, "Graph");
                match button_state
                {
                    TestButtonState::TestingDone => {
                    },
                    TestButtonState::AbortTest => {
                        if test_button_response.clicked()
                        {
                            if let Some(abort) = &self.test_nn_abort_tx
                            {
                                if let Err(e) = abort.send(())
                                {
                                    log::error!("Failed to send abort test signal: {:?}", e);
                                }
                            }
                        }
                    },
                    TestButtonState::StartTest => {
                        if test_button_response.clicked() {
                            if let Some(training_data) = *state_ctx.test_button_training_data {
                                if training_data.test_split_len() >= 1 {
                                    let (tx,rx) = mpsc::channel();
                                    let (tx_abort, rx_abort) = mpsc::channel();
                                    // if graph disabled, turn off sending data
                                    let maybe_tx = if self.test_nn_graph
                                    {
                                        Some(tx)
                                    }else{
                                        None
                                    };

                                    state_ctx.payload_test_buffer.clear();
                                    self.test_nn_abort_tx = Some(tx_abort);
                                    self.test_nn_rx = Some(rx);
                                    self.test_nn_done = Arc::new(Mutex::new(None));
                                    let test_nn_done_clone = self.test_nn_done.clone();
                                    let is_correct_fn = state_ctx.ai_is_corret_fn.clone();
                                    let training_data_clone = training_data.clone();
                                    self.test_nn_handle = Some(std::thread::spawn(move || {
                                        log::trace!("Test Thread test_nn spawned!");
                                        let pingpong = &mut NeuralNetworkPingPong::new(ai_clone.max_layer_nodes());
                                        match test_nn(&mut ai_clone, &training_data_clone.test_split(), is_correct_fn, maybe_tx, Some(rx_abort), pingpong)
                                        {
                                            Ok(r) => {
                                                log::trace!("Test Thread test_nn complete!");
                                                let save_path = "testresults.results";
                                                log::trace!("Saving results... {save_path}");
                                                if let Err(e) = r.save_results(save_path)
                                                {
                                                    log::error!("Failed to save results to {save_path}: {e}");
                                                }
                                                *test_nn_done_clone.lock().unwrap() = Some(r.clone());
                                            },
                                                Err(e) => {log::error!("{e}");
                                                *test_nn_done_clone.lock().unwrap() = Some(TestResults::new(vec![], ConfusionEvaluator::LargestLabel, 0.0));
                                            },
                                        }
                                    }));
                                    
                                } else {
                                    log::error!(
                                        "Could not start test, training data training len was empty"
                                    );
                                }
                            } else {
                                log::error!("Training dataset not set, could not train");
                            }
                        }
                    }
                };
            });
            } else {
                ui.label("NN not set");
            }
        })
    }
}

impl WindowAi {}


#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(PartialEq, Clone)]
pub struct AiSetupPreset{
    graph: GraphStructure, 
    dropout_prob: LayerTypeCPU, 
    softmax_output: bool,
    activation_func: ActivationFunctionType,
    is_correct_fn: ConfusionEvaluator, 
    cost_fn: CostFunction, 
    weight_init: WeightInit, 
    bias_init: BiasInit,
    display: String, 
}

impl std::fmt::Display for AiSetupPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}", self.display
        )
    }
}

use std::sync::LazyLock;
pub static MNIST_PRESET: LazyLock<AiSetupPreset> = LazyLock::new(|| {
    AiSetupPreset {
        graph: GraphStructure {
            input_nodes: 784,
            hidden_layers: vec![256, 128],
            output_nodes: 10,
        },
        dropout_prob: 0.3,
        softmax_output: true,
        activation_func: ActivationFunctionType::ReLU,
        is_correct_fn: ConfusionEvaluator::LargestLabel,
        cost_fn: CostFunction::CrossEntropyMulticlass,
        weight_init: WeightInit::HeUniform,
        bias_init: BiasInit::ZeroPointZeroOne,
        display: "MNIST_PRESET".to_string(),
    }
});
pub static ZLBL_PRESET: LazyLock<AiSetupPreset> = LazyLock::new(|| {
    AiSetupPreset {
        graph: GraphStructure {
            input_nodes: 4096,
            hidden_layers: vec![1024, 512],
            output_nodes: 2,
        },
        dropout_prob: 0.3,
        softmax_output: false,
        activation_func: ActivationFunctionType::Sigmoid,
        is_correct_fn: ConfusionEvaluator::Zlbl,
        cost_fn: CostFunction::CrossEntropyBinary,
        weight_init: WeightInit::XavierUniform,
        bias_init: BiasInit::ZeroPointZeroOne,
        display: "ZLBL_PRESET".to_string(),
    }
});
pub static ALL_PRESETS: LazyLock<[&'static AiSetupPreset; 2]> = LazyLock::new(|| {
    [&*MNIST_PRESET, &*ZLBL_PRESET]
});

pub struct WindowAiSetupPresetsCtx<'a> {
    pub window_data: &'a mut MenuWindowData,
    pub state: &'a mut AppState,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowAiSetupPresets {
    pub cached_ai_preset: AiSetupPreset
}

impl Default for WindowAiSetupPresets
{
    fn default() -> Self {
        Self { cached_ai_preset: ALL_PRESETS[0].clone() }
    }
}

impl<'a> DrawableWindow<'a> for WindowAiSetupPresets {
    type Ctx = WindowAiSetupPresetsCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        egui::Window::new("Setup Presets").default_pos(egui::pos2(0.0, 500.0)).show(ctx, |ui| {
           
            // TODO: use &AiSetupPreset instead of AiSetupPreset to avoid clones.
            let _before = self.cached_ai_preset.clone();
            let _combo_response = egui::ComboBox::from_label("AiSetup")
                .selected_text(self.cached_ai_preset.to_string())
                .show_ui(ui, |ui| {
                    for variant in *ALL_PRESETS {
                        ui.selectable_value(
                            &mut self.cached_ai_preset,
                            variant.clone(),
                            variant.to_string(),
                        );
                    }
                });
            // let changed_preset = before != self.cached_ai_preset;

            if ui.button("Setup").clicked()
            {
                state_ctx.window_data.graph_structure_string = self.cached_ai_preset.graph.to_string();
                state_ctx.window_data.ai_use_softmax_output = self.cached_ai_preset.softmax_output;
                state_ctx.window_data.ai_activation_function = self.cached_ai_preset.activation_func;
                state_ctx.window_data.ai_cost_fn= self.cached_ai_preset.cost_fn;
                state_ctx.window_data.ai_dropout_prob = self.cached_ai_preset.dropout_prob;
                state_ctx.window_data.ai_is_correct_fn = self.cached_ai_preset.is_correct_fn;
                state_ctx.window_data.ai_weight_init = self.cached_ai_preset.weight_init;
                state_ctx.window_data.ai_bias_init = self.cached_ai_preset.bias_init;

                *state_ctx.state = AppState::SetupAi;
            }
        })
    }
}

impl WindowAiSetupPresets {}

pub struct WindowTrainingSetCtx<'a> {
    pub training_data: &'a mut TrainingData, // Probably should store on heap to avoid copy, not an issue for now
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowTrainingSet {
    ui_training_dataset_split_thresholds_0: f64,
    ui_training_dataset_split_thresholds_1: f64,
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_zaoai_loader: Option<ZaoaiLabelsLoader>,
    #[cfg_attr(feature = "serde", serde(skip))]
    resize_text: String,
    cached_resize_input_dim: Vec<usize>,
}

impl Default for WindowTrainingSet {
    fn default() -> Self {
        Self {
            ui_training_dataset_split_thresholds_0: 1.0,
            ui_training_dataset_split_thresholds_1: 1.0,
            cached_zaoai_loader: None,
            resize_text: String::new(),
            cached_resize_input_dim: Vec::with_capacity(2),
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
        let response = egui::Window::new("Dataset")
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
                state_ctx.training_data.training_split_len(),
                100.0 * state_ctx.training_data.get_thresholds()[0],
                state_ctx.training_data.validation_split_len(),
                100.0 * (state_ctx.training_data.get_thresholds()[1] - state_ctx.training_data.get_thresholds()[0]),
                state_ctx.training_data.test_split_len(),
                100.0 * (1.0 - state_ctx.training_data.get_thresholds()[1]),
                state_ctx.training_data.len(),
                (state_ctx.training_data.training_split_len() + state_ctx.training_data.validation_split_len()
                    + state_ctx.training_data.test_split_len()) as f64
                    / state_ctx.training_data.len().max(1) as f64,
                ));

                ui.label(format!("Dimensions: ({}, {})", state_ctx.training_data.get_in_out_dimensions().0, state_ctx.training_data.get_in_out_dimensions().1));
                if ui.button("Load [2, 2] test dataset").clicked()
                {
                    let dataset = create_2x2_test_datapoints(0, 100000 as i32);
                    *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new(&dataset));
                    self.ui_training_dataset_split_thresholds_0 = state_ctx.training_data.get_thresholds()[0];
                    self.ui_training_dataset_split_thresholds_1 = state_ctx.training_data.get_thresholds()[1];
                }
                if ui.button("Load [784, 10] MNIST dataset").clicked() {
                let mnist = get_mnist();
                
                let map_mnist = |(image, &label): (&[u8; 784], &u8)| -> DataPoint {
                    let inputs: Vec<LayerTypeCPU> = image.iter().map(|&p| p as LayerTypeCPU / 255.0).collect();
                    let mut expected_outputs = vec![0.0; 10];
                    if (label as usize) < 10 {
                        expected_outputs[label as usize] = 1.0;
                    }
                    DataPoint { inputs, expected_outputs }
                };

                let dataset_train: Vec<DataPoint> = mnist.train_data.iter().zip(mnist.train_labels.iter()).map(map_mnist).collect();
                let dataset_test: Vec<DataPoint> = mnist.train_data.iter().zip(mnist.train_labels.iter()).map(map_mnist).collect();
                
                *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new_from_splits(&dataset_train, &vec![], &dataset_test));
                
                let thresholds = state_ctx.training_data.get_thresholds();
                self.ui_training_dataset_split_thresholds_0 = thresholds[0];
                self.ui_training_dataset_split_thresholds_1 = thresholds[1];
            }

                let button_text = self.cached_zaoai_loader.as_ref()
                    .and_then(|loader| loader.label_input_dim.map(|d| format!("Load [{}*{}, {}] spectrogram test", d[0], d[1], 2)))
                    .unwrap_or_else(|| format!("Load [{}, {}] spectrogram test", SPECTROGRAM_WIDTH * SPECTROGRAM_HEIGHT, 2));
                if ui.add(egui::Button::new(button_text).sense(Sense::empty())).clicked()
                {
                    log::error!("This does not work. Spectrogram crate sucks and can not work with it easily...");
                    let path = "test_files/test0.mkv";
                    let spectrogram = generate_spectrogram(&PathBuf::from(path), S_SPECTROGRAM_NUM_BINS);
                    match spectrogram
                    {
                        Ok(o) => { let new_point = AnimeDataPoint {
                        path: PathBuf::from(path),
                        spectrogram: o,
                        expected_outputs: vec![0.08936, 0.1510],
                    };

                    let dims = unsafe { get_spectrogram_dims(&new_point.spectrogram) };
                    assert_eq!(SPECTROGRAM_WIDTH, dims.0);
                    assert_eq!(SPECTROGRAM_HEIGHT, dims.1);
                    let dataset: Vec<_> = vec![DataPoint::from_anime_data_point(new_point, dims.0, dims.1)];
                    *state_ctx.training_data = TrainingData::Physical(TrainingDataset::new(&dataset));
                    state_ctx.training_data.set_thresholds(1.0, 1.0);},
                        Err(e) => log::error!("{:?}", e),
                    }
                }

                // let zaoai_label_path  = "training_data\\firstoutputlabels\\zaoai_labels";
                let zaoai_label_path  = "training_data\\output\\zaoai_labels";
                if self.cached_zaoai_loader.is_none()
                {
                    let new_label_loader = ZaoaiLabelsLoader::new(&zaoai_label_path);
                    match new_label_loader
                    {
                        Ok(ok) => {
                            let culled = ok
                             .cull_by(|a| a.expected_outputs().is_some());
                            self.cached_zaoai_loader = Some(culled);
                            log::info!("Set self.cached_zaoai_loader!: {}", self.cached_zaoai_loader.as_mut().unwrap().len());
                        }
                        Err(e) => log::error!("Failed to load ZaoaiLabelsLoader: {:?}", e),
                    }
                }
                if let  Some(zaoai_label_loader) = &self.cached_zaoai_loader 
                {
                    ui.horizontal(|ui|
                    {
                        if ui.button(format!("Load [{}, {}] {} ZaoaiLabels", SPECTROGRAM_WIDTH*SPECTROGRAM_HEIGHT, 2, zaoai_label_loader.len())).clicked()
                        {
                            let zaoai_labels = zaoai_label_loader.load_zaoai_labels().expect("failed to load zaoai_labels");
                            *state_ctx.training_data = TrainingData::Virtual(VirtualTrainingDataset::new(PathBuf::from(zaoai_label_path), zaoai_labels, [SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT]));
                        }

                        let name_label = ui.label("Resize");
                        if ui.text_edit_singleline(&mut self.resize_text).labelled_by(name_label.id).lost_focus() {
                            self.cached_resize_input_dim = self.resize_text
                                .split(|c| c == ',' || c == ' ')
                                .filter_map(|s| s.parse().ok())
                                .collect();
                        }
                    });
                }

            });

        if self.cached_resize_input_dim.len() >= 2
        {
            if state_ctx.training_data.get_in_out_dimensions().0 != (self.cached_resize_input_dim[0] * self.cached_resize_input_dim[1])
            {
                match state_ctx.training_data
                {
                    TrainingData::Physical(_training_dataset) => {},
                    TrainingData::Virtual(virtual_training_dataset) => {
                        log::info!("Set virtual trainingdata desiered dim: [{},{}]", self.cached_resize_input_dim[0], self.cached_resize_input_dim[1]);
                        virtual_training_dataset.set_desiered_input_dim([self.cached_resize_input_dim[0], self.cached_resize_input_dim[1]]);
                    },
                }
            }
        }

        response
    }
}

pub struct WindowTrainingSessionCtx<'a> {
    pub training_session: &'a mut TrainingSession,
    pub app_state: &'a mut AppState,
    pub training_thread: &'a mut TrainingThreadController,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WindowTrainingSession {
}

impl<'a> DrawableWindow<'a> for WindowTrainingSession {
    type Ctx = WindowTrainingSessionCtx<'a>;

    fn draw_ui(
        &mut self,
        ctx: &egui::Context,
        state_ctx: &mut Self::Ctx,
    ) -> Option<InnerResponse<Option<()>>> {
        egui::Window::new("Training")
            .default_pos(egui::pos2(350.0, 0.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.num_epochs,
                            RangeInclusive::new(1, 100),
                        )
                        .step_by(1.0)
                        .clamping(egui::SliderClamping::Never),
                    );
                    ui.label("Num Epochs");
                });

                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.batch_size,
                            RangeInclusive::new(10, 1000),
                        )
                        .step_by(1.0)
                        .clamping(egui::SliderClamping::Never),
                    );
                    ui.label("Batch Size");
                });

                ui.horizontal(|ui| {
                    add_slider_sized(
                        ui,
                        100.0,
                        Slider::new(
                            &mut state_ctx.training_session.learn_rate,
                            RangeInclusive::new(0.01, 0.5),
                        )
                        .clamping(egui::SliderClamping::Never)
                        .min_decimals(2)
                        .max_decimals_opt(Some(5)),
                    );
                    ui.label("Learn Rate");
                });
                
                let before = state_ctx.training_session.learn_rate_decay.clone();
                let text_none = "None";
                let _combo_response = egui::ComboBox::from_label("Decay")
                    .selected_text(state_ctx.training_session.learn_rate_decay.as_ref().and_then(|f|Some(f.to_string())).unwrap_or(text_none.to_string()))
                    .show_ui(ui, |ui| {
                        for variant in [
                                None,
                                Some(FloatDecay::Exponential { rate: 0.05 }),
                                Some(FloatDecay::StepDecay {
                                    step_size: 1,
                                    decay_factor: 0.5,
                                }),
                                Some(FloatDecay::Linear {
                                    max_steps: state_ctx.training_session.num_epochs,
                                    end_rate: 0.001,
                                }),
                                Some(FloatDecay::Cosine {
                                    max_steps: state_ctx.training_session.num_epochs,
                                    min_val: 0.001,
                                }),
                        ] {
                            ui.selectable_value(
                                &mut state_ctx.training_session.learn_rate_decay,
                                variant.clone(),
                                variant.and_then(|f|Some(f.to_string())).unwrap_or(text_none.to_string()),
                            );
                        }
                    });
                let _changed = before != state_ctx.training_session.learn_rate_decay;

                state_ctx.training_session.learn_rate_decay.as_mut().and_then(|f|{f.set_max_steps(state_ctx.training_session.num_epochs); Some(f)});
                let decay = &state_ctx.training_session.learn_rate_decay;
                

                let slider_enabled = decay.as_ref().map_or(false, |d| d.uses_decay_rate());

                ui.horizontal(|ui| {
                    let mut decay_rate = state_ctx.training_session.learn_rate_decay_rate;

                    let slider = ui.add_enabled(
                        slider_enabled,
                        Slider::new(&mut decay_rate, 0.01..=1.0)
                            .clamping(SliderClamping::Always)
                            .min_decimals(2)
                            .max_decimals_opt(Some(5)),
                    );

                    if slider.changed() {
                        state_ctx.training_session.learn_rate_decay_rate = decay_rate;

                        if let Some(ref mut decay) = state_ctx.training_session.learn_rate_decay {
                            if decay.uses_decay_rate() {
                                decay.set_decay_rate(decay_rate);
                            }
                        }
                    }

                    ui.label("Learn Decay Rate");
                });

                ui.horizontal(|ui| {
                    let _slider = ui.add(Slider::new(&mut state_ctx.training_session.validation_each_epoch,0..=5)
                            .clamping(egui::SliderClamping::Never)
                            .integer());

                    ui.label("Validate each epoch");
                });


                if *state_ctx.app_state == AppState::Training {
                    if ui.button("Abort Training").clicked() {
                        log::info!("Interupting Training");
                        if let Err(e) = state_ctx.training_thread.send_abort_training()
                        {
                            log::error!("Failed to send abort training signal: {:?}", e);
                        }
                        state_ctx.training_session.set_state(TrainingState::Finish);
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
            y: cost as f64,
        };
        result.push(plotpoint);
    }
    result
}
pub fn generate_last_loss_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let learn_rate = payload.training_metadata.last_loss;
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: learn_rate as f64, 
        };
        result.push(plotpoint);
    }
    result
}

pub fn generate_learn_rate_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let learn_rate = payload.training_metadata.learn_rate;
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: learn_rate as f64, 
        };
        result.push(plotpoint);
    }
    result
}
pub fn generate_f1_score_plotpoints_from_training_thread_payloads(
    payloads: &Vec<TrainingThreadPayload>,
) -> Vec<PlotPoint> {
    let mut result: Vec<PlotPoint> = Vec::with_capacity(payloads.len());

    for payload in payloads {
        let learn_rate = payload.training_metadata.calc_f1_score();
        let plotpoint = PlotPoint {
            x: payload.payload_index as f64,
            y: learn_rate as f64, 
        };
        result.push(plotpoint);
    }
    result
}
