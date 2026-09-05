// ============================
// Activation Functions
// ============================

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use zaoai_types::ai_labels::LayerTypeCPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Default, Clone, Copy, bincode::Encode, bincode::Decode, PartialEq)]
pub enum ActivationFunctionType {
    #[default]
    ReLU,
    Sigmoid,
    Softmax,
    Linear,
}

impl ActivationFunctionType {
    // Used by simd
    pub fn activate_scalar(&self, x: LayerTypeCPU) -> LayerTypeCPU {
        match self {
            ActivationFunctionType::ReLU => relu(x),
            ActivationFunctionType::Sigmoid => sigmoid(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax needs full vector context, use apply_softmax()")
            }
            ActivationFunctionType::Linear => x,
        }
    }
    // Used by simd
    pub fn activate_derivative_scalar(&self, x: LayerTypeCPU) -> LayerTypeCPU {
        match self {
            ActivationFunctionType::ReLU => relu_d(x),
            ActivationFunctionType::Sigmoid => sigmoid_d(x),
            ActivationFunctionType::Softmax => {
                unreachable!("Softmax derivative needs vector context")
            }
            ActivationFunctionType::Linear => 1.0,
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
                ActivationFunctionType::Linear => "Linear",
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
