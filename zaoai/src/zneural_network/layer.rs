use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use wide::f32x8;

// ============================
// Activation Functions
// ============================

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
        // Optimized SIMD version — mock example
        // You'd use something like `faster`, `wide`, `packed_simd`, or `std::simd`

        let len = layer_values.len();
        let mut output = vec![0.0f32; len];

        let max_val = layer_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // SIMD chunking
        let mut sum = 0.0f64;
        let chunks = layer_values.chunks_exact(8);
        let remainder = chunks.remainder();

        for chunk in chunks {
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

        let mut sum = 0.0f64;
        for &value in layer_values {
            sum += (value - max_val).exp() as f64;
        }

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

// ============================
// Cost Functions
// ============================

pub fn node_cost(output_activation: f32, expected_activation: f32) -> f32 {
    let error = output_activation - expected_activation;
    0.5 * error * error
}

pub fn node_cost_d(output_activation: f32, expected_activation: f32) -> f32 {
    (output_activation - expected_activation)
}

#[cfg(feature = "simd")]
pub fn node_cost_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    let error = output_activation - expected_activation;
    // 0.5 * error^2
    f32x8::splat(0.5) * error * error
}

#[cfg(feature = "simd")]
pub fn node_cost_d_simd(output_activation: f32x8, expected_activation: f32x8) -> f32x8 {
    output_activation - expected_activation
}
// ============================

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct Layer {
    pub num_in_nodes: usize,
    pub num_out_nodes: usize,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub weights_cost_grads: Vec<Vec<f32>>,
    pub biases_cost_grads: Vec<f32>,
    pub activation_type: ActivationFunctionType,
    pub dropout_prob: Option<f32>,
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct LayerLearnData {
    pub inputs: Vec<f32>,
    pub weighted_inputs: Vec<f32>,
    pub activation_values: Vec<f32>,
    //"node values" for the output layer. This is an array containing for each node:
    // the partial derivative of the cost with respect to the weighted input
    pub node_values: Vec<f32>,
    pub dropout_mask: Option<Vec<f32>>, // same length as layer outputs
}

impl Layer {
    pub fn new(
        num_in_nodes: usize,
        num_out_nodes: usize,
        activation_type: ActivationFunctionType,
        dropout_prob: Option<f32>,
    ) -> Layer {
        // Validate Inputs
        if num_in_nodes <= 0 {
            panic!(
                "NumInNodes must be > 0, got [{} {}]",
                num_in_nodes, num_out_nodes
            );
        }
        if num_out_nodes <= 0 {
            panic!(
                "NumOutNodes must be > 0, got [{} {}]",
                num_in_nodes, num_out_nodes
            );
        }

        // Allocate memory
        // Bias
        let mut biases = vec![0.0; num_out_nodes];
        let mut biases_cost_grads: Vec<f32> = Vec::new();
        biases_cost_grads.resize(num_out_nodes, 0.0);

        // Weight
        let mut weights: Vec<Vec<f32>> = Vec::new();
        weights.reserve(num_out_nodes);
        for i in 0..num_out_nodes {
            weights.push(vec![0.0; num_in_nodes]);
        }
        let mut weights_cost_grads: Vec<Vec<f32>> = Vec::new();
        weights_cost_grads.reserve(num_out_nodes);
        for i in 0..num_out_nodes {
            weights_cost_grads.push(vec![0.0; num_in_nodes]);
        }

        let mut activation_values: Vec<f32> = vec![0.0; num_in_nodes];
        let mut weighted_inputs: Vec<f32> = vec![0.0; num_out_nodes];

        let mut new_layer = Layer {
            num_in_nodes,
            num_out_nodes,
            weights,
            biases,
            weights_cost_grads,
            biases_cost_grads,
            activation_type,
            dropout_prob,
        };

        new_layer.init_weights_and_biases(0);

        new_layer
    }

    fn init_weights_and_biases(&mut self, seed: u64) {
        // Uniform [0-1]
        // {
        //     let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        //     let min_weight = 0.0;
        //     let max_weight = 1.0;
        //     let min_bias = min_weight;
        //     let max_bias = max_weight;

        //     // Initialize weights & biases
        //     for i in 0..self.num_out_nodes {
        //         let rand_bias: f32 = rng.gen_range(min_weight..max_weight);
        //         self.biases[i] = rand_bias;

        //         for j in 0..self.num_in_nodes {
        //             let rand_weight: f32 = rng.gen_range(min_bias..max_bias);
        //             self.weights[i][j] = rand_weight;
        //         }
        //     }
        // }
        // Xavier uniform
        {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

            let limit = (6.0 / (self.num_in_nodes + self.num_out_nodes) as f32).sqrt();

            for i in 0..self.num_out_nodes {
                self.biases[i] = 0.0; // biases often initialized to zero

                for j in 0..self.num_in_nodes {
                    self.weights[i][j] = rng.gen_range(-limit..limit);
                }
            }
        }
    }

    fn compute_weighted_inputs_scalar(&self, inputs: &[f32], output_buf: &mut [f32]) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        for out_i in 0..self.num_out_nodes {
            let mut sum = self.biases[out_i];
            let weights_row = &self.weights[out_i];
            for in_i in 0..self.num_in_nodes {
                sum += inputs[in_i] * weights_row[in_i];
            }
            output_buf[out_i] = sum;
        }
    }

    #[cfg(feature = "simd")]
    fn compute_weighted_inputs_simd(&self, inputs: &[f32], output_buf: &mut [f32]) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        const CHUNK_SIZE: usize = 8;
        let chunks = self.num_in_nodes / CHUNK_SIZE;
        let remainder = self.num_in_nodes % CHUNK_SIZE;

        for out_i in 0..self.num_out_nodes {
            let weights_row = &self.weights[out_i];
            let mut sum = f32x8::splat(0.0);

            for i in 0..chunks {
                let offset = i * CHUNK_SIZE;
                let a = f32x8::from(&inputs[offset..offset + CHUNK_SIZE]);
                let b = f32x8::from(&weights_row[offset..offset + CHUNK_SIZE]);
                sum += a * b;
            }

            let mut weighted_sum = sum.reduce_add();

            for i in (self.num_in_nodes - remainder)..self.num_in_nodes {
                weighted_sum += inputs[i] * weights_row[i];
            }

            output_buf[out_i] = weighted_sum + self.biases[out_i];
        }
    }

    #[cfg(feature = "simd")]
    fn compute_weighted_inputs_simd_rayon(&self, inputs: &[f32], output_buf: &mut [f32]) {
        use rayon::prelude::*;

        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        const CHUNK_SIZE: usize = 8;
        let chunks = self.num_in_nodes / CHUNK_SIZE;
        let remainder = self.num_in_nodes % CHUNK_SIZE;

        let weights = &self.weights;
        let biases = &self.biases;

        output_buf
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_i, out_val)| {
                let weight_row = &weights[out_i];
                let mut sum = f32x8::splat(0.0);

                for i in 0..chunks {
                    let offset = i * CHUNK_SIZE;
                    let a = f32x8::from(&inputs[offset..offset + CHUNK_SIZE]);
                    let b = f32x8::from(&weight_row[offset..offset + CHUNK_SIZE]);
                    sum += a * b;
                }

                let mut weighted_sum = sum.reduce_add();

                for i in (inputs.len() - remainder)..inputs.len() {
                    weighted_sum += inputs[i] * weight_row[i];
                }

                *out_val = weighted_sum + biases[out_i];
            });
    }

    #[cfg(not(feature = "simd"))]
    pub fn apply_activation(weighted_inputs: &[f32], t: ActivationFunctionType) -> Vec<f32> {
        weighted_inputs.iter().map(|&x| t.activate(x)).collect()
    }

    #[cfg(feature = "simd")]
    pub fn apply_activation(input: &[f32], t: ActivationFunctionType) -> Vec<f32> {
        const CHUNK_SIZE: usize = 8;

        let mut result = Vec::with_capacity(input.len());

        let chunks = input.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            let input_vec = f32x8::from(chunk);

            let activated_vec = t.activate_simd(input_vec);

            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            result.extend_from_slice(&out);
        }

        for &x in remainder {
            result.push(x.max(0.0)); // Or a scalar fallback version of your activation function
        }

        result
    }

    #[cfg(not(feature = "simd"))]
    fn fill_learn_data(&self, learn_data: &mut LayerLearnData, weighted_inputs: &[f32]) {
        assert_eq!(learn_data.weighted_inputs.len(), self.num_out_nodes);
        assert_eq!(learn_data.activation_values.len(), self.num_out_nodes);

        learn_data.weighted_inputs.copy_from_slice(weighted_inputs);

        for (w_in, act) in learn_data
            .weighted_inputs
            .iter()
            .zip(learn_data.activation_values.iter_mut())
        {
            *act = self.activation_type.activate(*w_in);
        }
    }

    // not tested
    #[cfg(feature = "simd")]
    fn fill_learn_data(&self, learn_data: &mut LayerLearnData, weighted_inputs: &[f32]) {
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

        // Process full chunks with SIMD activate_simd
        for chunk in chunks {
            let input_vec = f32x8::from(chunk);
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            learn_data.activation_values.extend_from_slice(&out);
        }

        // Process remainder with SIMD (zero-padded)
        if !remainder.is_empty() {
            // Create an array of zeros first
            let mut padded = [0.0f32; CHUNK_SIZE];
            // Copy remainder slice into beginning of padded
            padded[..remainder.len()].copy_from_slice(remainder);
            // Load into SIMD vector
            let input_vec = f32x8::from(padded);
            // Activate SIMD
            let activated_vec = self.activation_type.activate_simd(input_vec);
            let out: [f32; CHUNK_SIZE] = activated_vec.into();
            // Copy only the valid elements (remainder.len())
            learn_data
                .activation_values
                .extend_from_slice(&out[..remainder.len()]);
        }
    }

    pub fn calculate_outputs(&self, inputs: &[f32]) -> Vec<f32> {
        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_scalar(inputs, &mut weighted_inputs);
        Self::apply_activation(&weighted_inputs, self.activation_type)
    }

    #[cfg(feature = "simd")]
    pub fn calculate_outputs_simd(&self, inputs: &[f32]) -> Vec<f32> {
        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_simd(inputs, &mut weighted_inputs);
        Self::apply_activation(&weighted_inputs, self.activation_type)
    }

    #[cfg(feature = "simd")]
    pub fn calculate_outputs_simd_rayon(&self, inputs: &[f32]) -> Vec<f32> {
        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_simd_rayon(inputs, &mut weighted_inputs);
        Self::apply_activation(&weighted_inputs, self.activation_type)
    }

    pub fn calculate_outputs_learn(
        &mut self,
        learn_data: &mut LayerLearnData,
        inputs: &[f32],
    ) -> Vec<f32> {
        learn_data.inputs.clone_from_slice(inputs);

        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_scalar(inputs, &mut weighted_inputs);

        self.fill_learn_data(learn_data, &weighted_inputs);
        learn_data.activation_values.clone()
    }

    #[cfg(feature = "simd")]
    pub fn calculate_outputs_learn_simd(
        &mut self,
        learn_data: &mut LayerLearnData,
        inputs: &[f32],
    ) -> Vec<f32> {
        learn_data.inputs.clone_from_slice(inputs);

        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_simd(inputs, &mut weighted_inputs);

        self.fill_learn_data(learn_data, &weighted_inputs);
        learn_data.activation_values.clone()
    }

    #[cfg(feature = "simd")]
    pub fn calculate_outputs_learn_simd_rayon(
        &mut self,
        learn_data: &mut LayerLearnData,
        inputs: &[f32],
    ) -> Vec<f32> {
        learn_data.inputs.clone_from_slice(inputs);

        let mut weighted_inputs = vec![0.0; self.num_out_nodes];
        self.compute_weighted_inputs_simd_rayon(inputs, &mut weighted_inputs);

        self.fill_learn_data(learn_data, &weighted_inputs);
        learn_data.activation_values.clone()
    }

    pub fn apply_cost_gradient(&mut self, learn_rate: f32) {
        for node_out in 0..self.num_out_nodes {
            self.biases[node_out] -= self.biases_cost_grads[node_out] * learn_rate;

            for node_in in 0..self.num_in_nodes {
                self.weights[node_out][node_in] -=
                    self.weights_cost_grads[node_out][node_in] * learn_rate;
            }
        }
    }

    fn update_cost_gradient_for_node(
        weight_grad_row: &mut [f32],
        bias_grad: &mut f32,
        node_value: f32,
        inputs: &[f32],
        num_in_nodes: usize,
    ) {
        for node_in in 0..num_in_nodes {
            let derivative_cost_weight = inputs[node_in] * node_value;
            weight_grad_row[node_in] += derivative_cost_weight;
        }
        *bias_grad += node_value; // same as 1.0 * node_value
    }

    #[cfg(feature = "simd")]
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

        // SIMD chunks
        for i in 0..chunks {
            let offset = i * CHUNK_SIZE;
            let input_vec = f32x8::from(&inputs[offset..offset + CHUNK_SIZE]);
            let mut grad_vec = f32x8::from(&weight_grad_row[offset..offset + CHUNK_SIZE]);
            grad_vec += input_vec * node_value_vec;
            weight_grad_row[offset..offset + CHUNK_SIZE].copy_from_slice(&grad_vec.to_array());
        }

        // Scalar tail
        for i in (num_in_nodes - remainder)..num_in_nodes {
            weight_grad_row[i] += inputs[i] * node_value;
        }

        // Update bias gradient
        *bias_grad += node_value;
    }

    pub fn update_cost_gradients(&mut self, learn_data: &LayerLearnData) {
        let num_in_nodes = self.num_in_nodes;
        let inputs = &learn_data.inputs;

        for node_out in 0..self.num_out_nodes {
            if let Some(mask) = learn_data.dropout_mask.as_ref() {
                if mask[node_out] == 0.0 {
                    continue; // neuron was dropped, skip gradient update
                }
            }

            let node_value = learn_data.node_values[node_out];
            let weight_grad_row = &mut self.weights_cost_grads[node_out];
            let bias_grad = &mut self.biases_cost_grads[node_out];

            Self::update_cost_gradient_for_node(
                weight_grad_row,
                bias_grad,
                node_value,
                inputs,
                num_in_nodes,
            );
        }
    }

    #[cfg(feature = "simd")]
    pub fn update_cost_gradients_simd(&mut self, learn_data: &LayerLearnData) {
        let num_in_nodes = self.num_in_nodes;
        let inputs = &learn_data.inputs;

        for node_out in 0..self.num_out_nodes {
            if let Some(mask) = learn_data.dropout_mask.as_ref() {
                if mask[node_out] == 0.0 {
                    continue; // neuron was dropped, skip gradient update
                }
            }

            let node_value = learn_data.node_values[node_out];
            let weight_grad_row = &mut self.weights_cost_grads[node_out];
            let bias_grad = &mut self.biases_cost_grads[node_out];

            Self::update_cost_gradient_for_node_simd(
                weight_grad_row,
                bias_grad,
                node_value,
                inputs,
                num_in_nodes,
            );
        }
    }

    #[cfg(feature = "simd")]
    pub fn update_cost_gradients_simd_rayon(&mut self, learn_data: &LayerLearnData) {
        let num_in_nodes = self.num_in_nodes;
        let inputs = &learn_data.inputs;
        let maybe_mask = learn_data.dropout_mask.as_ref();

        self.weights_cost_grads
            .par_iter_mut()
            .zip(self.biases_cost_grads.par_iter_mut())
            .zip(learn_data.node_values.par_iter().copied())
            .enumerate() // <- Add this to get index
            .for_each(|(node_out, ((weight_grad_row, bias_grad), node_value))| {
                if let Some(mask) = maybe_mask {
                    if mask[node_out] == 0.0 {
                        return; // neuron was dropped, skip gradient update
                    }
                }

                Self::update_cost_gradient_for_node_simd(
                    weight_grad_row,
                    bias_grad,
                    node_value,
                    inputs,
                    num_in_nodes,
                );
            });
    }

    pub fn clear_cost_gradient(&mut self) {
        self.biases_cost_grads.fill(0.0);
        for row in &mut self.weights_cost_grads {
            row.fill(0.0);
        }
    }

    #[cfg(not(feature = "simd"))]
    pub fn calculate_output_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[f32],
    ) {
        for i in 0..learn_data.node_values.len() {
            let dcost = node_cost_d(learn_data.activation_values[i], expected_outputs[i]);
            let dactivation = self
                .activation_type
                .activate_derivative(learn_data.weighted_inputs[i]);
            learn_data.node_values[i] = dactivation * dcost;
        }
    }

    #[cfg(feature = "simd")]
    pub fn calculate_output_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        expected_outputs: &[f32],
    ) {
        use wide::f32x8;

        const CHUNK_SIZE: usize = 8;
        let len = learn_data.node_values.len();

        let activation_vals = &learn_data.activation_values;
        let weighted_inputs = &learn_data.weighted_inputs;
        let node_vals = &mut learn_data.node_values;

        let chunks_activation = activation_vals.chunks_exact(CHUNK_SIZE);
        let chunks_weighted = weighted_inputs.chunks_exact(CHUNK_SIZE);
        let chunks_expected = expected_outputs.chunks_exact(CHUNK_SIZE);
        let mut chunks_node_vals = node_vals.chunks_exact_mut(CHUNK_SIZE);

        // Save remainders *before* consuming the iterators
        let remainder_activation = chunks_activation.remainder();
        let remainder_weighted = chunks_weighted.remainder();
        let remainder_expected = chunks_expected.remainder();
        // For chunks_node_vals, we have to consume it before calling into_remainder()
        // So no remainder here yet

        // Use .by_ref() to iterate chunks_node_vals without moving it
        for ((chunk_activation, chunk_weighted), (chunk_expected, chunk_node_vals)) in
            chunks_activation
                .zip(chunks_weighted)
                .zip(chunks_expected.zip(chunks_node_vals.by_ref()))
        {
            let act_vec = f32x8::from(chunk_activation);
            let weighted_vec = f32x8::from(chunk_weighted);
            let expected_vec = f32x8::from(chunk_expected);

            let dcost = node_cost_d_simd(act_vec, expected_vec);
            let dactivation = self.activation_type.activate_derivative_simd(weighted_vec);

            let result = dactivation * dcost;

            let result_arr: [f32; CHUNK_SIZE] = result.into();
            chunk_node_vals.copy_from_slice(&result_arr);
        }

        // Now get the remainder from chunks_node_vals *after* the loop
        let remainder_node_vals = chunks_node_vals.into_remainder();

        // Handle remainder scalars (fallback)
        if !remainder_activation.is_empty() {
            for i in 0..remainder_activation.len() {
                let dcost = node_cost_d(remainder_activation[i], remainder_expected[i]);
                let dactivation = self
                    .activation_type
                    .activate_derivative(remainder_weighted[i]);
                remainder_node_vals[i] = dactivation * dcost;
            }
        }
    }

    pub fn calculate_hidden_layer_node_cost_values(
        &self,
        learn_data: &mut LayerLearnData,
        prev_layer: &Layer,
        prev_node_cost_values: &[f32],
    ) {
        for new_node_index in 0..self.num_out_nodes {
            let mut new_node_value: f32 = 0.0;
            for prev_node_index in 0..prev_node_cost_values.len() {
                let weighted_input_d = prev_layer.weights[prev_node_index][new_node_index];
                new_node_value += weighted_input_d * prev_node_cost_values[prev_node_index];
            }

            new_node_value *= self
                .activation_type
                .activate_derivative(learn_data.weighted_inputs[new_node_index]);
            learn_data.node_values[new_node_index] = new_node_value;
        }
    }
}
