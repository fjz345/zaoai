// ============================
// Activation Functions
// ============================

#[cfg(feature = "simd")]
use crate::simd::*;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "simd")]
use wide::f32x8;
use zaoai_types::ai_labels::LayerTypeCPU;

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
    pub fn apply_softmax(layer_values: &mut [f32]) {
        let max_val = layer_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f64;
        let chunks = layer_values.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            use wide::f32x8;

            // TODO: as_mut_chunk
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
        for val in layer_values.iter_mut() {
            *val = (*val - max_val).exp() / sum_f32;
        }
    }
    #[cfg(not(feature = "simd"))]
    pub fn apply_softmax(layer_values: &[LayerTypeCPU]) -> Vec<LayerTypeCPU> {
        let max_val = layer_values
            .iter()
            .cloned()
            .fold(LayerTypeCPU::NEG_INFINITY, LayerTypeCPU::max);
        let sum: LayerTypeCPU = layer_values
            .iter()
            .map(|&v| (v - max_val).exp() as LayerTypeCPU)
            .sum();

        layer_values
            .iter()
            .map(|&v| ((v - max_val).exp() as LayerTypeCPU / sum) as LayerTypeCPU)
            .collect()
    }
    pub fn activate(&self, x: LayerTypeCPU) -> LayerTypeCPU {
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
    pub fn activate_derivative(&self, x: LayerTypeCPU) -> LayerTypeCPU {
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

fn sigmoid(in_value: LayerTypeCPU) -> LayerTypeCPU {
    1.0 / (1.0 + (-in_value).exp())
}

fn sigmoid_d(in_value: LayerTypeCPU) -> LayerTypeCPU {
    let f = sigmoid(in_value);
    f * (1.0 - f)
}

fn relu(in_value: LayerTypeCPU) -> LayerTypeCPU {
    in_value.max(0.0)
}
pub(crate) fn relu_d(in_value: LayerTypeCPU) -> LayerTypeCPU {
    if in_value > 0.0 {
        1.0
    } else {
        0.0
    }
}

// ============================
