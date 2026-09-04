use serde::{Deserialize, Serialize};
use strum_macros::Display;

use zaoai_types::ai_labels::LayerTypeCPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Display, bincode::Encode, bincode::Decode)]
pub enum CostFunction {
    Mse,
    CrossEntropyBinary,
    CrossEntropyMulticlass,
}

impl CostFunction {
    pub fn call(&self, predicted: &[LayerTypeCPU], expected: &[LayerTypeCPU]) -> LayerTypeCPU {
        match self {
            CostFunction::Mse => mse(predicted, expected),
            CostFunction::CrossEntropyBinary => cross_entropy_loss_binary(predicted, expected),
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass(predicted, expected)
            }
        }
    }
    pub fn call_d(&self, predicted: &[LayerTypeCPU], expected: &[LayerTypeCPU]) -> LayerTypeCPU {
        match self {
            CostFunction::Mse => mse_d(predicted, expected),
            CostFunction::CrossEntropyBinary => cross_entropy_loss_binary_d(predicted, expected),
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass_d(predicted, expected)
            }
        }
    }
}

// ============================
// Cost Functions
// ============================

pub fn mse_single(
    output_activation: LayerTypeCPU,
    expected_activation: LayerTypeCPU,
) -> LayerTypeCPU {
    let error = output_activation - expected_activation;
    0.5 * error * error
}
pub fn mse_single_d(
    output_activation: LayerTypeCPU,
    expected_activation: LayerTypeCPU,
) -> LayerTypeCPU {
    output_activation - expected_activation
}

pub fn mse(predicted: &[LayerTypeCPU], expected: &[LayerTypeCPU]) -> LayerTypeCPU {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| mse_single(*p, *e))
        .sum()
}
pub fn mse_d(predicted: &[LayerTypeCPU], expected: &[LayerTypeCPU]) -> LayerTypeCPU {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| mse_single_d(*p, *e))
        .sum()
}

pub fn cross_entropy_loss_multiclass(
    predicted: &[LayerTypeCPU],
    expected: &[LayerTypeCPU],
) -> LayerTypeCPU {
    let epsilon = 1e-12;

    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| {
            let p_clamped = p.max(epsilon).min(1.0 - epsilon);
            -e * p_clamped.ln()
        })
        .sum()
}
pub fn cross_entropy_loss_multiclass_d(
    predicted: &[LayerTypeCPU],
    expected: &[LayerTypeCPU],
) -> LayerTypeCPU {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, y)| p - y)
        .sum()
}
pub fn cross_entropy_loss_binary(
    predicted: &[LayerTypeCPU],
    expected: &[LayerTypeCPU],
) -> LayerTypeCPU {
    let epsilon = 1e-12;

    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| {
            let p_clamped = p.max(epsilon).min(1.0 - epsilon);
            -(e * p_clamped.ln() + (1.0 - e) * (1.0 - p_clamped).ln())
        })
        .sum()
}
pub fn cross_entropy_loss_binary_d(
    predicted: &[LayerTypeCPU],
    expected: &[LayerTypeCPU],
) -> LayerTypeCPU {
    let epsilon = 1e-12;
    let mut result = 0.0;

    for (&p, &y) in predicted.iter().zip(expected.iter()) {
        let p_clamped = p.max(epsilon).min(1.0 - epsilon);
        result += -y / p_clamped + (1.0 - y) / (1.0 - p_clamped);
    }

    result
}
// ============================
