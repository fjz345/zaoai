use eframe::egui::{
    self,
    plot::{GridInput, GridMark, Line, Plot, PlotPoint, PlotPoints},
};

pub struct TrainingGraphVisualization {
    pub(crate) title: String,
    pub(crate) should_show: bool,
    pub(crate) plot_data: Vec<PlotPoint>,
}

impl TrainingGraphVisualization {
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
