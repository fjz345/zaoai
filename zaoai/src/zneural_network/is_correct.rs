use serde::{Deserialize, Serialize};
use strum_macros::Display;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Display)]
pub enum IsCorrectFn {
    MaxVal,
    Zlbl,
    ZlblLoose,
}

impl IsCorrectFn {
    // Dispatch method: calls the appropriate function based on variant
    pub fn call(&self, outputs: &[f32], expected_outputs: &[f32]) -> bool {
        match self {
            IsCorrectFn::Zlbl => Self::zlbl_is_correct_fn(outputs, expected_outputs),
            IsCorrectFn::MaxVal => Self::largest_label_is_correct_fn(outputs, expected_outputs),
            IsCorrectFn::ZlblLoose => Self::zlbl_loose_is_correct_fn(outputs, expected_outputs),
        }
    }

    fn is_normalized_within_tolerance(
        predicted_normalized: f32,
        expected_normalized: f32,
        tolerance_seconds: f32,
        total_duration: std::time::Duration,
    ) -> bool {
        let epsilon = tolerance_seconds / total_duration.as_secs_f32();
        (predicted_normalized - expected_normalized).abs() <= epsilon
    }

    fn zlbl_is_correct_fn(outputs: &[f32], expected_outputs: &[f32]) -> bool {
        const EPSILON: f32 = 0.001;
        let duration = std::time::Duration::from_secs(20 * 60);
        outputs
            .iter()
            .zip(expected_outputs)
            .all(|(&output, &expected_output)| {
                Self::is_normalized_within_tolerance(output, expected_output, 2.0, duration)
            })
    }

    fn zlbl_loose_is_correct_fn(outputs: &[f32], expected_outputs: &[f32]) -> bool {
        const EPSILON: f32 = 0.001;
        let duration = std::time::Duration::from_secs(20 * 60);
        outputs
            .iter()
            .zip(expected_outputs)
            .all(|(&output, &expected_output)| {
                Self::is_normalized_within_tolerance(output, expected_output, 20.0, duration)
            })
    }

    fn largest_label_is_correct_fn(outputs: &[f32], expected_outputs: &[f32]) -> bool {
        let (determined_index, _) = Self::determine_output_greatest_value_result(outputs);
        let (expected_index, _) = Self::determine_output_greatest_value_result(expected_outputs);
        determined_index == expected_index
    }

    fn determine_output_greatest_value_result(outputs: &[f32]) -> (usize, f32) {
        outputs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, &val)| (idx, val))
            .unwrap_or((0, 0.0))
    }
}
