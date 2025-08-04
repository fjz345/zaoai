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
            CostFunction::Mse => Self::mse(predicted, expected),
            CostFunction::CrossEntropyBinary => {
                Self::cross_entropy_loss_binary(predicted, expected)
            }
            CostFunction::CrossEntropyMulticlass => {
                Self::cross_entropy_loss_multiclass(predicted, expected)
            }
        }
    }

    fn mse(predicted: &[f32], expected: &[f32]) -> f32 {
        predicted
            .iter()
            .zip(expected.iter())
            .map(|(p, e)| {
                let error = p - e;
                0.5 * error * error
            })
            .sum()
    }

    fn cross_entropy_loss_binary(predicted: &[f32], expected: &[f32]) -> f32 {
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

    fn cross_entropy_loss_multiclass(predicted: &[f32], expected: &[f32]) -> f32 {
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
pub fn mse_single_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    let error = output_activation - expected_activation;
    // 0.5 * error^2
    f32x8::splat(0.5) * error * error
}

#[cfg(feature = "simd")]
pub fn mse_single_d_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    output_activation - expected_activation
}

// Todo  simd
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
    // Small epsilon to avoid log(0)
    let epsilon = 1e-12;

    predicted
        .iter()
        .zip(expected.iter())
        .map(|(p, e)| {
            // Clamp p to avoid log(0)
            let p_clamped = p.max(epsilon).min(1.0 - epsilon);
            -e * p_clamped.ln()
        })
        .sum()
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
// ============================
