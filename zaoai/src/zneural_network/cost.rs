use serde::{Deserialize, Serialize};
use strum_macros::Display;
#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, PartialEq, Display, bincode::Encode, bincode::Decode)]
pub enum CostFunction {
    Mse,
    CrossEntropyBinary,
    CrossEntropyMulticlass,
}

impl CostFunction {
    pub fn call(&self, predicted: &[f32], expected: &[f32]) -> f32 {
        match self {
            CostFunction::Mse => mse(predicted, expected),
            CostFunction::CrossEntropyBinary => cross_entropy_loss_binary(predicted, expected),
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass(predicted, expected)
            }
        }
    }

    pub fn call_d(&self, predicted: &[f32], expected: &[f32]) -> f32 {
        match self {
            CostFunction::Mse => mse(predicted, expected),
            CostFunction::CrossEntropyBinary => cross_entropy_loss_binary_d(predicted, expected),
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass(predicted, expected)
            }
        }
    }

    pub fn call_simd(&self, predicted: f32x8, expected: f32x8) -> f32x8 {
        match self {
            CostFunction::Mse => mse_simd(predicted, expected),
            CostFunction::CrossEntropyBinary => cross_entropy_loss_binary_simd(predicted, expected),
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass_simd(predicted, expected)
            }
        }
    }

    pub fn call_simd_d(&self, predicted: f32x8, expected: f32x8) -> f32x8 {
        match self {
            CostFunction::Mse => mse_d_simd(predicted, expected),
            CostFunction::CrossEntropyBinary => {
                cross_entropy_loss_binary_d_simd(predicted, expected)
            }
            CostFunction::CrossEntropyMulticlass => {
                cross_entropy_loss_multiclass_d_simd(predicted, expected)
            }
        }
    }
}

// ============================
// Cost Functions
// ============================

pub fn mse_single(output_activation: f32, expected_activation: f32) -> f32 {
    let error = output_activation - expected_activation;
    0.5 * error * error
}

pub fn mse_single_d(output_activation: f32, expected_activation: f32) -> f32 {
    (output_activation - expected_activation)
}

#[cfg(feature = "simd")]
pub fn mse_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    let error = output_activation - expected_activation;
    // 0.5 * error^2
    f32x8::splat(0.5) * error * error
}

#[cfg(feature = "simd")]
pub fn mse_d_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    output_activation - expected_activation
}

pub fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| mse_single(*p, *e))
        .sum()
}

pub fn mse_d(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| mse_single_d(*p, *e))
        .sum()
}

pub fn cross_entropy_loss_multiclass(predicted: &[f32], expected: &[f32]) -> f32 {
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

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_multiclass_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);
    -expected * clamped.ln()
}

pub fn cross_entropy_loss_multiclass_d(predicted: &[f32], expected: &[f32]) -> f32 {
    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, y)| p - y)
        .sum()
}

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_multiclass_d_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    // Assumes inputs are after softmax
    predicted - expected
}

pub fn cross_entropy_loss_binary(predicted: &[f32], expected: &[f32]) -> f32 {
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

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_binary_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);

    -(expected * clamped.ln() + (one - expected) * (one - clamped).ln())
}

pub fn cross_entropy_loss_binary_d(predicted: &[f32], expected: &[f32]) -> f32 {
    let epsilon = 1e-12;
    let mut result = 0.0;

    for (&p, &y) in predicted.iter().zip(expected.iter()) {
        let p_clamped = p.max(epsilon).min(1.0 - epsilon);
        result += -y / p_clamped + (1.0 - y) / (1.0 - p_clamped);
    }

    result
}

#[cfg(feature = "simd")]
pub fn cross_entropy_loss_binary_d_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    // Clamp predicted to [epsilon, 1 - epsilon]
    let p = predicted.min(one - epsilon).max(epsilon);
    let one_minus_p = one - p;
    let one_minus_y = one - expected;

    // Derivative: - y / p + (1 - y) / (1 - p)
    -expected / p + one_minus_y / one_minus_p
}
// ============================
