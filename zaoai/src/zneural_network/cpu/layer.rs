use rand::prelude::*;
use rand_chacha::{self, ChaCha8Rng};
use rand_distr::{num_traits::FromPrimitive, Distribution, Normal};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;

use crate::weight_bias::{BiasInit, WeightInit};

use crate::zneural_network::activation::ActivationFunctionType;
use crate::zneural_network::cost::CostFunction;
use crate::zneural_network::cpu::neuralnetwork_cpu::NeuralNetworkPingPong;
use crate::zneural_network::datapoint::DataPoint;
use crate::zneural_network::weight_bias::WeightInitContext;
use zaoai_types::ai_labels::LayerTypeCPU;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct Layer {
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
    pub weights: Vec<Vec<LayerTypeCPU>>,
    pub biases: Vec<LayerTypeCPU>,
    pub weights_cost_grads: Vec<Vec<LayerTypeCPU>>,
    pub biases_cost_grads: Vec<LayerTypeCPU>,
    pub activation_type: ActivationFunctionType,
    pub dropout_prob: Option<LayerTypeCPU>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct LayerLearnData {
    pub inputs: Vec<LayerTypeCPU>,
    pub weighted_inputs: Vec<LayerTypeCPU>,
    pub activation_values: Vec<LayerTypeCPU>,
    //"node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub node_values: Vec<LayerTypeCPU>,
    pub dropout_mask: Option<Vec<LayerTypeCPU>>, // same length as layer outputs
}

impl Layer {
    pub fn new(
        num_in_nodes: usize,
        num_out_nodes: usize,
        activation_type: ActivationFunctionType,
        dropout_prob: Option<LayerTypeCPU>,
        weight_init: WeightInit,
        bias_init: BiasInit,
    ) -> Self {
        assert!(num_in_nodes > 0, "NumInNodes must be > 0");
        assert!(num_out_nodes > 0, "NumOutNodes must be > 0");

        // Initialize weights and gradients with zeros
        let weights = vec![vec![0.0; num_in_nodes]; num_out_nodes];
        let weights_cost_grads = vec![vec![0.0; num_in_nodes]; num_out_nodes];
        let biases = vec![0.0; num_out_nodes];
        let biases_cost_grads = vec![0.0; num_out_nodes];

        let mut layer = Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
            activation_type,
            dropout_prob,
        };

        layer.init_weights_and_biases(0, weight_init, bias_init);

        layer
    }

    pub fn init_weights_and_biases(
        &mut self,
        seed: u64,
        weight_init: WeightInit,
        bias_init: BiasInit,
    ) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let ctx = WeightInitContext::<LayerTypeCPU>::new(
            weight_init,
            self.num_in_nodes,
            self.num_out_nodes,
        );

        for i in 0..self.num_out_nodes {
            self.biases[i] = bias_init.sample_bias();

            for j in 0..self.num_in_nodes {
                self.weights[i][j] = ctx.sample_weight(&mut rng);
            }
        }
    }

    pub fn apply_activation(weighted_inputs: &mut [LayerTypeCPU], t: ActivationFunctionType) {
        #[cfg(feature = "simd")]
        Self::apply_activation_simd(weighted_inputs, t);
        #[cfg(not(feature = "simd"))]
        Self::apply_activation_scalar(weighted_inputs, t);
    }

    pub fn fill_learn_data(
        &self,
        learn_data: &mut LayerLearnData,
        weighted_inputs: &[LayerTypeCPU],
    ) {
        #[cfg(feature = "simd")]
        self.fill_learn_data_simd(learn_data, weighted_inputs);
        #[cfg(not(feature = "simd"))]
        self.fill_learn_data_scalar(learn_data, weighted_inputs);
    }

    pub fn calculate_outputs(&self, inputs: &[LayerTypeCPU], outputs: &mut [LayerTypeCPU]) {
        #[cfg(not(feature = "simd"))]
        self.calculate_outputs_scalar(inputs, outputs);
        #[cfg(feature = "simd")]
        self.calculate_outputs_simd(inputs, outputs);
    }

    pub fn calculate_outputs_learn(
        &mut self,
        inputs: &[LayerTypeCPU],
        outputs: &mut [LayerTypeCPU],
        learn_data: &mut LayerLearnData,
    ) {
        #[cfg(feature = "simd")]
        self.calculate_outputs_learn_simd(inputs, outputs, learn_data);
        #[cfg(not(feature = "simd"))]
        self.calculate_outputs_learn_scalar(inputs, outputs, learn_data);
    }

    #[inline]
    pub fn update_cost_gradients(&mut self, learn_data: &mut LayerLearnData) {
        #[cfg(feature = "simd")]
        self.update_cost_gradients_simd(learn_data);
        #[cfg(not(feature = "simd"))]
        self.update_cost_gradients_scalar(learn_data);
    }

    pub fn clear_cost_gradient(&mut self) {
        self.biases_cost_grads.fill(0.0);
        for row in &mut self.weights_cost_grads {
            row.fill(0.0);
        }
    }

    pub fn apply_cost_gradient(&mut self, learn_rate: LayerTypeCPU) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    pub fn calculate_hidden_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        prev_layer: &Layer,
        prev_node_cost_values: &[LayerTypeCPU],
    ) {
        learn_data.node_values.fill(0.0);

        for (prev_cost, weights_row) in prev_node_cost_values.iter().zip(prev_layer.weights.iter())
        {
            for (new_val, &weight) in learn_data.node_values.iter_mut().zip(weights_row.iter()) {
                *new_val += weight * prev_cost;
            }
        }
        for (new_val, &weighted_input) in learn_data
            .node_values
            .iter_mut()
            .zip(learn_data.weighted_inputs.iter())
        {
            *new_val *= self
                .activation_type
                .activate_derivative_scalar(weighted_input);
        }
    }
}

// All layers
pub fn forward(
    layers: &Vec<Layer>,
    inputs: &[LayerTypeCPU],
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) {
    pingpong.current.clear();
    pingpong.current.extend_from_slice(inputs);

    for layer in layers {
        pingpong.next.resize(layer.num_out_nodes, 0.0);
        layer.calculate_outputs(&pingpong.current, &mut pingpong.next);
        std::mem::swap(&mut pingpong.current, &mut pingpong.next);
    }

    // Simplicity, keep input at current, output at next
    std::mem::swap(&mut pingpong.current, &mut pingpong.next);

    if is_softmax_output {
        #[cfg(feature = "simd")]
        ActivationFunctionType::apply_softmax_simd(&mut pingpong.next);
        #[cfg(not(feature = "simd"))]
        ActivationFunctionType::apply_softmax_scalar(&mut pingpong.next);
    }
}

pub fn calculate_cost(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    #[cfg(not(feature = "simd"))]
    {
        use crate::zneural_network::cpu::scalar::calculate_cost_scalar;
        calculate_cost_scalar(layers, data, cost_fn, pingpong, is_softmax_output)
    }
    #[cfg(feature = "simd")]
    {
        use crate::zneural_network::cpu::simd::calculate_cost_simd;
        calculate_cost_simd(layers, data, cost_fn, pingpong, is_softmax_output)
    }
}
