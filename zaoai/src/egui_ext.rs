use std::ops::RangeInclusive;

use crate::egui::Shape::Path;
use eframe::egui::{
    Color32, DragValue, Pos2, Response, Sense, SliderOrientation, Stroke, TextStyle, Ui, Visuals,
    Widget,
};
use eframe::emath::NumExt;
use eframe::epaint::{vec2, PathShape, Rect, Vec2};
use eframe::{egui, emath};

pub fn add_slider_sized(ui: &mut egui::Ui, size: f32, slider: egui::Slider) -> Response {
    let saved_slider_width = ui.style_mut().spacing.slider_width;
    ui.style_mut().spacing.slider_width = size;
    let result: Response = ui.add(slider);
    ui.style_mut().spacing.slider_width = saved_slider_width;
    result
}

fn make_triangle(pos: Pos2, size: f64) -> Vec<Pos2> {
    vec![
        Pos2 {
            x: pos.x - size as f32 / 2.0,
            y: pos.y + size as f32 / 2.0,
        },
        Pos2 {
            x: pos.x,
            y: pos.y - size as f32 / 2.0,
        },
        Pos2 {
            x: pos.x + size as f32 / 2.0,
            y: pos.y + size as f32 / 2.0,
        },
    ]
}

/// Combined into one function (rather than two) to make it easier
/// for the borrow checker.
type GetSetValue<'a> = Box<dyn 'a + FnMut(Option<f64>) -> f64>;
fn get(get_set_value: &mut GetSetValue<'_>) -> f64 {
    (get_set_value)(None)
}

fn set(get_set_value: &mut GetSetValue<'_>, value: f64) {
    (get_set_value)(Some(value));
}

pub struct Interval<'a> {
    get_set_value_0: GetSetValue<'a>, // upper bound
    get_set_value_1: GetSetValue<'a>, // lower bound
    range: RangeInclusive<f64>,
    clamp_to_range: bool,
    show_value: bool,
    orientation: SliderOrientation,
    // text: WidgetText,
    /// Sets the minimal step of the widget value
    step: Option<f64>,
    // drag_value_speed: Option<f64>,
    // min_decimals: usize,
    // max_decimals: Option<usize>,
}

impl<'a> Interval<'a> {
    fn add_contents(&mut self, ui: &mut Ui) -> Response {
        let thickness = ui
            .text_style_height(&TextStyle::Body)
            .at_least(ui.spacing().interact_size.y);
        let response = self.allocate_slider_space(ui, thickness);
        let triangle_color: Color32;
        if ui.visuals() == &Visuals::dark() {
            triangle_color = Color32::WHITE;
        } else {
            triangle_color = Color32::BLACK;
        }
        let rect = response.rect;
        if ui.is_rect_visible(rect) {
            let mut lb = get(&mut self.get_set_value_0);
            let mut ub = get(&mut self.get_set_value_1);
            let range = self.range.end() - self.range.start();
            let bar_line = vec![rect.left_center(), rect.right_center()];
            ui.painter().add(Path(PathShape::line(
                bar_line,
                Stroke::new(rect.height() / 5.0 as f32, Color32::GRAY),
            )));
            let mut bounds_i = [
                lb / range * rect.width() as f64,
                ub / range * rect.width() as f64,
            ];
            let mut lb_pos = Pos2 {
                x: rect.left() + bounds_i[0] as f32,
                y: rect.left_center().y,
            };
            let mut ub_pos = Pos2 {
                x: rect.left() + bounds_i[1] as f32,
                y: rect.right_center().y,
            };

            if response.dragged() {
                let pos = response.interact_pointer_pos();

                const TRIANGLE_SNAP_DISTANCE: f32 = 25.0;
                match pos {
                    None => {}
                    Some(p) => {
                        let lb_dist_to_p = lb_pos.distance(p);
                        let ub_dist_to_p = ub_pos.distance(p);
                        if lb_dist_to_p <= TRIANGLE_SNAP_DISTANCE
                            || ub_dist_to_p <= TRIANGLE_SNAP_DISTANCE
                        {
                            if (lb_dist_to_p <= ub_dist_to_p) {
                                // dragging the lower one
                                lb_pos.x = p.x.min(ub_pos.x);
                            } else {
                                // dragging the upper one
                                ub_pos.x = p.x.max(lb_pos.x);
                            }
                        }
                    }
                }
            }

            bounds_i[0] = lb_pos.x as f64 - rect.left() as f64;
            lb = bounds_i[0] / rect.width() as f64 * range;
            bounds_i[1] = ub_pos.x as f64 - rect.left() as f64;
            ub = bounds_i[1] / rect.width() as f64 * range;

            let interval_line = vec![lb_pos, ub_pos];
            ui.painter().add(Path(PathShape::line(
                interval_line,
                Stroke::new(rect.height() / 4.0 as f32, Color32::LIGHT_GRAY),
            )));
            ui.painter().add(Path(PathShape::convex_polygon(
                make_triangle(lb_pos, rect.height() as f64 / 2.0),
                triangle_color,
                Stroke::new(1.0, triangle_color),
            )));
            ui.painter().add(Path(PathShape::convex_polygon(
                make_triangle(ub_pos, rect.height() as f64 / 2.0),
                triangle_color,
                Stroke::new(1.0, triangle_color),
            )));

            ui.horizontal(|ui| {
                ui.add(
                    DragValue::new(&mut lb)
                        .speed(0.01)
                        .range(0.0..=range)
                        .fixed_decimals(2),
                );
                ui.add(
                    DragValue::new(&mut ub)
                        .speed(0.01)
                        .range(0.0..=range)
                        .fixed_decimals(2),
                );
            });

            let lb_tmp = lb.clone();
            let ub_tmp = ub.clone();
            lb = lb_tmp.min(ub_tmp);
            ub = lb_tmp.max(ub_tmp);
            set(&mut self.get_set_value_0, lb);
            set(&mut self.get_set_value_1, ub);
        }
        response
    }

    /// Creates a new horizontal slider.
    pub fn new<Num: emath::Numeric>(
        value_0: &'a mut Num,
        value_1: &'a mut Num,
        range: RangeInclusive<Num>,
    ) -> Self {
        let range_f64 = range.start().to_f64()..=range.end().to_f64();
        let slf = Self::from_get_set(
            range_f64,
            move |v: Option<f64>| {
                if let Some(v) = v {
                    *value_0 = Num::from_f64(v);
                }
                value_0.to_f64()
            },
            move |v: Option<f64>| {
                if let Some(v) = v {
                    *value_1 = Num::from_f64(v);
                }
                value_1.to_f64()
            },
        );

        if Num::INTEGRAL {
            // slf.integer()
            slf
        } else {
            slf
        }
    }

    pub fn from_get_set(
        range: RangeInclusive<f64>,
        get_set_value_0: impl 'a + FnMut(Option<f64>) -> f64,
        get_set_value_1: impl 'a + FnMut(Option<f64>) -> f64,
    ) -> Self {
        Self {
            get_set_value_0: Box::new(get_set_value_0),
            get_set_value_1: Box::new(get_set_value_1),
            range,
            clamp_to_range: true,
            show_value: true,
            orientation: SliderOrientation::Horizontal,
            step: None,
        }
    }

    /// Just the slider, no text
    fn allocate_slider_space(&self, ui: &mut Ui, thickness: f32) -> (Response) {
        let desired_size = match self.orientation {
            SliderOrientation::Horizontal => vec2(ui.spacing().slider_width, thickness),
            SliderOrientation::Vertical => vec2(thickness, ui.spacing().slider_width),
        };
        ui.allocate_response(desired_size, Sense::drag())
    }
}

impl<'a> Widget for Interval<'a> {
    fn ui(mut self, ui: &mut Ui) -> Response {
        let inner_response = match self.orientation {
            SliderOrientation::Horizontal => ui.horizontal(|ui| self.add_contents(ui)),
            SliderOrientation::Vertical => ui.vertical(|ui| self.add_contents(ui)),
        };

        inner_response.inner | inner_response.response
    }
}
