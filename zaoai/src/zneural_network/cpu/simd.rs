use crate::zneural_network::activation::relu_d;
use crate::zneural_network::cpu::layer::{forward, LayerLearnData};
use crate::zneural_network::cpu::neuralnetwork_cpu::NeuralNetworkPingPong;
use crate::zneural_network::datapoint::DataPoint;
use wide::f32x8;
use zaoai_types::ai_labels::LayerTypeCPU;

use crate::cost::CostFunction;
use crate::cpu::layer::Layer;

// ============================
// Cost Functions
// ============================

impl CostFunction {
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

pub fn mse_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    let error = output_activation - expected_activation;
    // 0.5 * error^2
    f32x8::splat(0.5) * error * error
}
pub fn mse_d_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    output_activation - expected_activation
}

pub fn cross_entropy_loss_multiclass_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);
    -expected * clamped.ln()
}
pub fn cross_entropy_loss_multiclass_d_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    // Assumes inputs are after softmax
    predicted - expected
}
pub fn cross_entropy_loss_binary_simd(predicted: f32x8, expected: f32x8) -> f32x8 {
    let epsilon = f32x8::splat(1e-12);
    let one = f32x8::splat(1.0);

    let clamped = predicted.min(one - epsilon).max(epsilon);

    -(expected * clamped.ln() + (one - expected) * (one - clamped).ln())
}
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

// ============================
// Activation Functions
// ============================
pub fn sigmoid_simd(x: f32x8) -> f32x8 {
    let one = f32x8::splat(1.0);
    one / (one + (-x).exp())
}
pub fn sigmoid_d_simd(x: f32x8) -> f32x8 {
    let fx = sigmoid_simd(x);
    fx * (f32x8::splat(1.0) - fx)
}
pub fn relu_simd(in_value: f32x8) -> f32x8 {
    // Fastmax?
    in_value.max(f32x8::splat(0.0))
}
pub fn relu_d_simd(x: f32x8) -> f32x8 {
    // temp fix
    let a: Vec<f32> = x.to_array().iter_mut().map(|f| relu_d(*f)).collect();
    return f32x8::from(&a[..]);

    // think this is correct?, no was wrong...
    // use wide::CmpGt;
    // x.cmp_gt(f32x8::splat(0.0))
}
// ============================

use crate::activation::ActivationFunctionType;

impl ActivationFunctionType {
    pub fn apply_softmax_simd(layer_values: &mut [f32]) {
        let max_val = layer_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f64;
        let chunks = layer_values.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let v = f32x8::from(chunk);
            let e = (v - f32x8::splat(max_val)).exp();

            for val in e.to_array() {
                sum += val as f64;
            }
        }

        for &val in remainder {
            sum += (val - max_val).exp() as f64;
        }

        let sum_f32 = sum as f32;

        for val in layer_values.iter_mut() {
            *val = (*val - max_val).exp() / sum_f32;
        }
    }

    pub fn activate_simd(&self, x: f32x8) -> f32x8 {
        match self {
            Self::ReLU => relu_simd(x),
            Self::Sigmoid => sigmoid_simd(x),
            Self::Softmax => {
                unreachable!("Softmax needs full vector context, use apply_softmax()")
            }
            Self::Linear => x,
        }
    }

    pub fn activate_derivative_simd(&self, x: f32x8) -> f32x8 {
        match self {
            Self::ReLU => relu_d_simd(x),
            Self::Sigmoid => sigmoid_d_simd(x),
            Self::Softmax => {
                unreachable!("Softmax derivative needs vector context")
            }
            Self::Linear => f32x8::splat(1.0),
        }
    }
}

impl Layer {
    pub fn compute_weighted_inputs_simd(&self, inputs: &[f32], output_buf: &mut [f32]) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        for ((output, weights_row), &bias) in output_buf
            .iter_mut()
            .zip(self.weights.iter())
            .zip(self.biases.iter())
        {
            let mut sum = f32x8::splat(0.0);
            let input_chunks = inputs.as_chunks::<8>();
            let weight_chunks = weights_row.as_chunks::<8>();

            for (i_chunk, w_chunk) in input_chunks.0.iter().zip(weight_chunks.0.iter()) {
                sum += f32x8::from(*i_chunk) * f32x8::from(*w_chunk);
            }

            let mut weighted_sum = sum.reduce_add();

            for (i, w) in input_chunks.1.iter().zip(weight_chunks.1.iter()) {
                weighted_sum += i * w;
            }

            *output = weighted_sum + bias;
        }
    }

    pub fn fill_learn_data_simd(&self, learn_data: &mut LayerLearnData, weighted_inputs: &[f32]) {
        use wide::f32x8;

        assert_eq!(learn_data.weighted_inputs.len(), self.num_out_nodes);
        assert_eq!(learn_data.activation_values.len(), self.num_out_nodes);

        learn_data.weighted_inputs.copy_from_slice(weighted_inputs);

        const CHUNK_SIZE: usize = 8;

        let len = learn_data.weighted_inputs.len();
        let chunks = learn_data.weighted_inputs.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        learn_data.activation_values.clear();
        learn_data.activation_values.reserve(len);

        for chunk in chunks {
            let input_vec = f32x8::from(chunk);
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            learn_data.activation_values.extend_from_slice(&out);
        }

        if !remainder.is_empty() {
            let mut padded = [0.0f32; CHUNK_SIZE];
            padded[..remainder.len()].copy_from_slice(remainder);
            let input_vec = f32x8::from(padded);
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            // Copy only the valid elements (remainder.len())
            learn_data
                .activation_values
                .extend_from_slice(&out[..remainder.len()]);
        }
    }

    pub fn apply_activation_simd(input: &mut [f32], t: ActivationFunctionType) {
        const CHUNK_SIZE: usize = 8;

        let mut chunks = input.chunks_exact_mut(CHUNK_SIZE);
        // let mut chunks = input.as_chunks_mut::<CHUNK_SIZE>().0;

        for chunk in chunks.by_ref() {
            let arr: [f32; CHUNK_SIZE] = (&*chunk).try_into().unwrap();
            let input_vec = f32x8::from(arr);

            let activated_vec = t.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();

            chunk.copy_from_slice(&out);
        }

        for x in chunks.into_remainder() {
            *x = t.activate_scalar(*x);
        }
    }

    pub fn calculate_outputs_simd(&self, inputs: &[f32], outputs: &mut [f32]) {
        self.compute_weighted_inputs_simd(inputs, outputs);
        Self::apply_activation(outputs, self.activation_type);
    }

    pub fn calculate_outputs_learn_simd(
        &mut self,
        inputs: &[f32],
        outputs: &mut [f32],
        learn_data: &mut LayerLearnData,
    ) {
        learn_data.inputs.clear();
        learn_data.inputs.extend_from_slice(inputs);

        self.compute_weighted_inputs_simd(inputs, outputs);
        self.fill_learn_data(learn_data, &outputs);
        outputs.copy_from_slice(&learn_data.activation_values);
    }

    fn update_cost_gradient_for_node_simd(
        weight_grad_row: &mut [f32],
        bias_grad: &mut f32,
        node_value: f32,
        inputs: &[f32],
        num_in_nodes: usize,
    ) {
        const CHUNK_SIZE: usize = 8;
        let chunks = num_in_nodes / CHUNK_SIZE;
        let remainder = num_in_nodes % CHUNK_SIZE;
        let node_value_vec = f32x8::splat(node_value);

        for i in 0..chunks {
            let offset = i * CHUNK_SIZE;
            let input_vec = f32x8::from(&inputs[offset..offset + CHUNK_SIZE]);
            let mut grad_vec = f32x8::from(&weight_grad_row[offset..offset + CHUNK_SIZE]);
            grad_vec += input_vec * node_value_vec;
            weight_grad_row[offset..offset + CHUNK_SIZE].copy_from_slice(&grad_vec.to_array());
        }

        for i in (num_in_nodes - remainder)..num_in_nodes {
            weight_grad_row[i] += inputs[i] * node_value;
        }
        // Update bias gradient
        *bias_grad += node_value;
    }

    pub fn update_cost_gradients_simd(&mut self, learn_data: &LayerLearnData) {
        let inputs = &learn_data.inputs;
        let num_in_nodes = self.num_in_nodes;

        if let Some(mask) = learn_data.dropout_mask.as_ref() {
            for (((weight_grad_row, bias_grad), &node_value), &m) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
                .zip(mask.iter())
            {
                if m != 0.0 {
                    Self::update_cost_gradient_for_node_simd(
                        weight_grad_row,
                        bias_grad,
                        node_value,
                        inputs,
                        num_in_nodes,
                    );
                }
            }
        } else {
            for ((weight_grad_row, bias_grad), &node_value) in self
                .weights_cost_grads
                .iter_mut()
                .zip(self.biases_cost_grads.iter_mut())
                .zip(learn_data.node_values.iter())
            {
                Self::update_cost_gradient_for_node_simd(
                    weight_grad_row,
                    bias_grad,
                    node_value,
                    inputs,
                    num_in_nodes,
                );
            }
        }
    }

    pub fn calculate_output_layer_node_cost_values_simd(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[f32],
        cost_fn: CostFunction,
    ) {
        use wide::f32x8;

        const CHUNK_SIZE: usize = 8;

        let activation_vals = &learn_data.activation_values;
        let weighted_inputs = &learn_data.weighted_inputs;
        let node_vals = &mut learn_data.node_values;

        let chunks_activation = activation_vals.chunks_exact(CHUNK_SIZE);
        let chunks_weighted = weighted_inputs.chunks_exact(CHUNK_SIZE);
        let chunks_expected = expected_outputs.chunks_exact(CHUNK_SIZE);
        let mut chunks_node_vals = node_vals.chunks_exact_mut(CHUNK_SIZE);

        let remainder_activation = chunks_activation.remainder();
        let remainder_weighted = chunks_weighted.remainder();
        let remainder_expected = chunks_expected.remainder();

        for ((chunk_activation, chunk_weighted), (chunk_expected, chunk_node_vals)) in
            chunks_activation
                .zip(chunks_weighted)
                .zip(chunks_expected.zip(chunks_node_vals.by_ref()))
        {
            let act_vec = f32x8::from(chunk_activation);
            let weighted_vec = f32x8::from(chunk_weighted);
            let expected_vec = f32x8::from(chunk_expected);

            let dcost = cost_fn.call_simd_d(act_vec, expected_vec);
            let dactivation = self.activation_type.activate_derivative_simd(weighted_vec);

            let result = dactivation * dcost;

            let result_arr: [f32; CHUNK_SIZE] = result.into();
            chunk_node_vals.copy_from_slice(&result_arr);
        }

        let remainder_node_vals = chunks_node_vals.into_remainder();

        if !remainder_activation.is_empty() {
            for i in 0..remainder_activation.len() {
                let dcost =
                    cost_fn.call_d(&vec![remainder_activation[i]], &vec![remainder_expected[i]]);
                let dactivation = self
                    .activation_type
                    .activate_derivative_scalar(remainder_weighted[i]);
                remainder_node_vals[i] = dactivation * dcost;
            }
        }
    }
}

pub fn calculate_cost_simd(
    layers: &Vec<Layer>,
    data: &[DataPoint],
    cost_fn: CostFunction,
    pingpong: &mut NeuralNetworkPingPong,
    is_softmax_output: bool,
) -> LayerTypeCPU {
    use zaoai_types::ai_labels::LayerTypeCPU;

    let num_outputs = layers.last().unwrap().num_out_nodes;

    let total_cost: LayerTypeCPU = data
        .iter()
        .map(|datapoint| {
            forward(layers, &datapoint.inputs, pingpong, is_softmax_output);
            let mut sum = f32x8::splat(0.0);
            let mut i = 0;

            while i + 8 <= num_outputs {
                let pred = f32x8::from(&pingpong.next[i..i + 8]);
                let expected = f32x8::from(&datapoint.expected_outputs[i..i + 8]);
                sum += cost_fn.call_simd(pred, expected);
                i += 8;
            }

            let mut cost = sum.reduce_add();
            if i < num_outputs {
                cost += cost_fn.call(&pingpong.next[i..], &datapoint.expected_outputs[i..]);
            }
            cost
        })
        .sum();

    let l2_penalty: LayerTypeCPU = layers
        .iter()
        .flat_map(|layer| layer.weights.iter())
        .flat_map(|matrix| matrix.iter())
        .map(|w| w.powi(2))
        .sum();

    (total_cost / (data.len() as LayerTypeCPU)) + (0.001 * l2_penalty)
}
