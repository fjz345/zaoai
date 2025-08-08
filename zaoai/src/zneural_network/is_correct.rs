use serde::{Deserialize, Serialize};
use strum_macros::Display;

use crate::zneural_network::datapoint::{self, DataPoint};

pub enum ConfusionCategory {
    TruePositive,
    TrueNegative,
    FalsePositive,
    FalseNegative,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Display)]
pub enum ConfusionEvaluator {
    BinaryThreshold { threshold: f32 },
    LargestLabel,
    Zlbl,
    ZlblLoose,
}

impl ConfusionEvaluator {
    pub fn evaluate(&self, predicted: &[f32], expected: &[f32]) -> ConfusionCategory {
        match self {
            ConfusionEvaluator::BinaryThreshold { threshold } => {
                let pred_label = predicted.get(0).copied().unwrap_or(0.0) > *threshold;
                let exp_label = expected.get(0).copied().unwrap_or(0.0) > 0.5;
                Self::binary_confusion_from_labels(exp_label, pred_label)
            }
            ConfusionEvaluator::LargestLabel => {
                let pred_idx = Self::determine_output_greatest_value_result(predicted).0;
                let exp_idx = Self::determine_output_greatest_value_result(expected).0;
                if exp_idx == pred_idx {
                    ConfusionCategory::TruePositive // or true negative? Depends on your definition.
                } else {
                    ConfusionCategory::FalsePositive // or false negative?
                }
            }
            ConfusionEvaluator::Zlbl => {
                if Self::zlbl_is_correct_fn(predicted, expected) {
                    ConfusionCategory::TruePositive
                } else {
                    ConfusionCategory::FalsePositive
                }
            }
            ConfusionEvaluator::ZlblLoose => {
                if Self::zlbl_loose_is_correct_fn(predicted, expected) {
                    ConfusionCategory::TruePositive
                } else {
                    ConfusionCategory::FalsePositive
                }
            }
        }
    }

    fn binary_confusion_from_labels(expected: bool, predicted: bool) -> ConfusionCategory {
        match (expected, predicted) {
            (true, true) => ConfusionCategory::TruePositive,
            (false, false) => ConfusionCategory::TrueNegative,
            (false, true) => ConfusionCategory::FalsePositive,
            (true, false) => ConfusionCategory::FalseNegative,
        }
    }

    fn determine_output_greatest_value_result(outputs: &[f32]) -> (usize, f32) {
        outputs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, &val)| (idx, val))
            .unwrap_or((0, 0.0))
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
        let duration = std::time::Duration::from_secs(20 * 60);
        outputs
            .iter()
            .zip(expected_outputs)
            .all(|(&output, &expected_output)| {
                Self::is_normalized_within_tolerance(output, expected_output, 2.0, duration)
            })
    }

    fn zlbl_loose_is_correct_fn(outputs: &[f32], expected_outputs: &[f32]) -> bool {
        let duration = std::time::Duration::from_secs(20 * 60);
        outputs
            .iter()
            .zip(expected_outputs)
            .all(|(&output, &expected_output)| {
                Self::is_normalized_within_tolerance(output, expected_output, 20.0, duration)
            })
    }
}
