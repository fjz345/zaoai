// #![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use std::str::FromStr;

use egui::plot::{Line, Plot, PlotPoints, PlotPoints::Owned};

use eframe::{
    egui::{self, plot::PlotPoint, style::Widgets, Response},
    epaint::Color32,
    App,
};
use graphviz_rust::dot_structures::Graph;

use crate::zneural_network::neuralnetwork::{GraphStructure, NeuralNetwork};

/// State per thread.
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

struct MenuWindowState {
    nn_structure: String,
    show_training_graph: bool,
}

pub struct ZaoaiApp {
    state: AppState,
    ai: Option<NeuralNetwork>,
    window_state: MenuWindowState,
    training_graph: TrainingGraphVisualization,
}

impl Default for ZaoaiApp {
    fn default() -> Self {
        Self {
            state: AppState::Startup,
            ai: None,
            window_state: MenuWindowState {
                nn_structure: "2, 3, 2".to_owned(),
                show_training_graph: true,
            },
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

    fn draw_ui_menu(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.checkbox(
                    &mut self.window_state.show_training_graph,
                    "Show Training Graph",
                );

                let name_label = ui.label("Create new NN with layers");
                if (ui
                    .text_edit_singleline(&mut self.window_state.nn_structure)
                    .labelled_by(name_label.id)
                    .lost_focus())
                {
                    self.state = AppState::SetupAi;
                }
            });
        });

        self.draw_ui_ai(ctx);

        self.training_graph.draw_ui(ctx);
    }
}

impl eframe::App for ZaoaiApp {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        // NOTE: a bright gray makes the shadows of the windows look weird.
        // We use a bit of transparency so that if the user switches on the
        // `transparent()` option they get immediate results.
        egui::Color32::from_rgba_unmultiplied(12, 12, 12, 180).to_normalized_gamma_f32()

        // _visuals.window_fill() would also be a natural choice
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        match self.state {
            AppState::Startup => {
                self.startup(ctx, frame);
                self.state = AppState::SetupAi;
            }
            AppState::SetupAi => {
                let mut formatted_nn_structure = self
                    .window_state
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
                        println!("ASD");
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
                self.draw_ui_menu(ctx, frame);

                if let Some(ai) = &self.ai {
                    let nn_structure: GraphStructure = GraphStructure::new(&[2, 3, 2], true);
                    let mut nntest: NeuralNetwork = NeuralNetwork::new(nn_structure);
                    nntest.validate();
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
        self.training_graph.should_show = self.window_state.show_training_graph;
    }
}

fn add_slider_sized(ui: &mut egui::Ui, size: f32, slider: egui::Slider) -> Response {
    let saved_slider_width = ui.style_mut().spacing.slider_width;
    ui.style_mut().spacing.slider_width = size;
    let result: Response = ui.add(slider);
    ui.style_mut().spacing.slider_width = saved_slider_width;
    result
}
