// ============================
// Activation Functions
// ============================

use serde::{Deserialize, Serialize};
#[cfg(feature = "simd")]
use wide::f32x8;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, Copy, bincode::Encode, bincode::Decode, PartialEq)]
pub enum ActivationFunctionType {
    #[default]
    ReLU,
    Sigmoid,
    Softmax,
}

impl ActivationFunctionType {
    #[cfg(feature = "simd")]
    pub fn apply_softmax(layer_values: &[f32]) -> Vec<f32> {
        let len = layer_values.len();
        let mut output = vec![0.0f32; len];

        let max_val = layer_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f64;
        let chunks = layer_values.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            use wide::f32x8;

            let v = f32x8::from(chunk);
            let e = (v - f32x8::splat(max_val)).exp();
            let temp = e.to_array();
            for val in temp {
                sum += val as f64;
            }
        }

        for &val in remainder {
            sum += (val - max_val).exp() as f64;
        }

        let sum_f32 = sum as f32;

        // Normalize
        for (i, &val) in layer_values.iter().enumerate() {
            output[i] = ((val - max_val).exp() / sum_f32);
        }

        output
    }

    #[cfg(not(feature = "simd"))]
    pub fn apply_softmax(layer_values: &[f32]) -> Vec<f32> {
        let max_val = layer_values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = layer_values
            .iter()
            .map(|&v| (v - max_val).exp() as f64)
            .sum();

        layer_values
            .iter()
            .map(|&v| ((v - max_val).exp() as f64 / sum) as f32)
            .collect()
    }
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            ActivationFunctionType::ReLU => relu(x),
            ActivationFunctionType::Sigmoid => sigmoid(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax needs full vector context, use apply_softmax()")
            }
        }
    }

    #[cfg(feature = "simd")]
    pub fn activate_simd(&self, x: f32x8) -> f32x8 {
        match self {
            ActivationFunctionType::ReLU => relu_simd(x),
            ActivationFunctionType::Sigmoid => sigmoid_simd(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax needs full vector context, use apply_softmax()")
            }
        }
    }

    pub fn activate_derivative(&self, x: f32) -> f32 {
        match self {
            ActivationFunctionType::ReLU => relu_d(x),
            ActivationFunctionType::Sigmoid => sigmoid_d(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax derivative needs vector context")
            }
        }
    }

    #[cfg(feature = "simd")]
    pub fn activate_derivative_simd(&self, x: f32x8) -> f32x8 {
        match self {
            ActivationFunctionType::ReLU => relu_d_simd(x),
            ActivationFunctionType::Sigmoid => sigmoid_d_simd(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax derivative needs vector context")
            }
        }
    }
}

impl std::fmt::Display for ActivationFunctionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ActivationFunctionType::ReLU => "ReLU",
                ActivationFunctionType::Sigmoid => "Sigmoid",
                ActivationFunctionType::Softmax => "Softmax",
                // ActivationFunctionType::Tanh => "Tanh",
                // ActivationFunctionType::LeakyReLU => "Leaky ReLU",
                // ActivationFunctionType::Softmax => "Softmax",
            }
        )
    }
}

fn sigmoid(in_value: f32) -> f32 {
    1.0 / (1.0 + (-in_value).exp())
}

fn sigmoid_d(in_value: f32) -> f32 {
    let f = sigmoid(in_value);
    f * (1.0 - f)
}

#[cfg(feature = "simd")]
fn sigmoid_simd(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + (-x).exp())
}

#[cfg(feature = "simd")]
fn sigmoid_d_simd(x: f32x8) -> f32x8 {
    let fx = sigmoid_simd(x);
    fx * (f32x8::splat(1.0) - fx)
}

fn relu(in_value: f32) -> f32 {
    in_value.max(0.0)
}

#[cfg(feature = "simd")]
fn relu_simd(in_value: f32x8) -> f32x8 {
    // Fastmax?
    in_value.max(f32x8::splat(0.0))
}

fn relu_d(in_value: f32) -> f32 {
    if in_value > 0.0 {
        1.0
    } else {
        0.0
    }
}

#[cfg(feature = "simd")]
fn relu_d_simd(x: f32x8) -> f32x8 {
    // temp fix
    let a: Vec<f32> = x.to_array().iter_mut().map(|f| relu_d(*f)).collect();
    return f32x8::from(&a[..]);

    // think this is correct?, no was wrong...
    // use wide::CmpGt;
    // x.cmp_gt(f32x8::splat(0.0))
}
// ============================
