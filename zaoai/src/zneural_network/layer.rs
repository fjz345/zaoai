use rand::prelude::*;
use rand::rngs::StdRng;
use rand_chacha::{self, ChaCha8Rng};
use rand_distr::{num_traits::FromPrimitive, Distribution, Normal, Uniform};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::Display;
use wide::f32x8;

use crate::zneural_network::activation::ActivationFunctionType;
#[cfg(feature = "simd")]
use crate::zneural_network::cost::CostFunction;

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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum WeightInit {
    Zero,       // Bad
    Uniform,    // Uniform [0, 1]
    NormalDist, // Normal(0, 1)
    #[default]
    XavierUniform, // sigmoid / tanh
    XavierNormal, // sigmoid / tanh
    HeUniform,  // ReLU / leaky ReLU
    HeNormal,   // ReLU / leaky ReLU
    LeCun,      // SELU / scaled tanh
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Copy, Display, PartialEq, Default)]
pub enum BiasInit {
    Zero,
    #[default]
    ZeroPointZeroOne,
    // Random,
}

pub struct WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: Distribution<T>,
{
    pub weight_init: WeightInit,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub normal_dist: Option<rand_distr::Normal<T>>,
    pub limit: Option<T>,
}

impl<T> WeightInitContext<T>
where
    T: rand_distr::num_traits::Float + FromPrimitive,
    rand_distr::StandardNormal: rand::distributions::Distribution<T>,
{
    #[inline(always)]
    fn to_t(x: f64) -> T {
        T::from_f64(x).expect("conversion from f64 to T failed")
    }

    pub fn new(weight_init: WeightInit, num_inputs: usize, num_outputs: usize) -> Self {
        let (normal_dist, limit) = match weight_init {
            WeightInit::NormalDist => (
                Some(Normal::new(Self::to_t(0.0), Self::to_t(1.0)).unwrap()),
                None,
            ),
            WeightInit::XavierUniform => {
                let limit = (Self::to_t(6.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (None, Some(limit))
            }
            WeightInit::XavierNormal => {
                let std_dev = (Self::to_t(2.0)
                    / (Self::to_t(num_inputs as f64) + Self::to_t(num_outputs as f64)))
                .sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::HeUniform => {
                let limit = (Self::to_t(6.0) / Self::to_t(num_inputs as f64)).sqrt();
                (None, Some(limit))
            }
            WeightInit::HeNormal => {
                let std_dev = (Self::to_t(2.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            WeightInit::LeCun => {
                let std_dev = (Self::to_t(1.0) / Self::to_t(num_inputs as f64)).sqrt();
                (Some(Normal::new(Self::to_t(0.0), std_dev).unwrap()), None)
            }
            _ => (None, None), // Zero and Uniform don't need precalc
        };

        Self {
            weight_init,
            num_inputs,
            num_outputs,
            normal_dist,
            limit,
        }
    }

    pub fn sample_weight(&self, rng: &mut ChaCha8Rng) -> T {
        match self.weight_init {
            WeightInit::Zero => Self::to_t(0.0),
            WeightInit::Uniform => {
                T::from_f64(rng.gen_range(0.0..1.0)).expect("Uniform range failed")
            }
            WeightInit::NormalDist
            | WeightInit::XavierNormal
            | WeightInit::HeNormal
            | WeightInit::LeCun => self.normal_dist.as_ref().unwrap().sample(rng),
            WeightInit::XavierUniform | WeightInit::HeUniform => {
                let limit = self.limit.unwrap();
                let val = rng.gen_range(-limit.to_f64().unwrap()..limit.to_f64().unwrap());
                T::from_f64(val).unwrap()
            }
        }
    }
}

impl WeightInit {
    pub fn all() -> &'static [Self] {
        use crate::WeightInit::*;
        &[
            Zero,
            Uniform,
            NormalDist,
            XavierUniform,
            XavierNormal,
            HeUniform,
            HeNormal,
            LeCun,
        ]
    }
}

impl BiasInit {
    pub fn all() -> &'static [Self] {
        use crate::BiasInit::*;
        &[Zero, ZeroPointZeroOne]
    }

    pub fn sample_bias(self) -> f32 {
        match self {
            Self::Zero => 0.0,
            Self::ZeroPointZeroOne => 0.01,
        }
    }
}

impl Layer {
    pub fn new(
        num_in_nodes: usize,
        num_out_nodes: usize,
        activation_type: ActivationFunctionType,
        dropout_prob: Option<f32>,
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
        let ctx = WeightInitContext::<f32>::new(weight_init, self.num_in_nodes, self.num_out_nodes);

        for i in 0..self.num_out_nodes {
            self.biases[i] = bias_init.sample_bias();

            for j in 0..self.num_in_nodes {
                self.weights[i][j] = ctx.sample_weight(&mut rng);
            }
        }
    }

    fn compute_weighted_inputs_scalar(&self, inputs: &[f32], output_buf: &mut [f32]) {
        assert_eq!(inputs.len(), self.num_in_nodes);
        assert_eq!(output_buf.len(), self.num_out_nodes);

        for (out_i, output) in output_buf.iter_mut().enumerate() {
            let weights_row = &self.weights[out_i];
            let sum = self.biases[out_i]
                + inputs
                    .iter()
                    .zip(weights_row.iter())
                    .map(|(input, weight)| input * weight)
                    .sum::<f32>();
            *output = sum;
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

            if remainder != 0 {
                for i in (self.num_in_nodes - remainder)..self.num_in_nodes {
                    weighted_sum += inputs[i] * weights_row[i];
                }
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
            result.push(t.activate(x));
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
        inputs: &[f32],
        learn_data: &mut LayerLearnData,
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
        inputs: &[f32],
        learn_data: &mut LayerLearnData,
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
        inputs: &[f32],
        learn_data: &mut LayerLearnData,
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
        cost_fn: CostFunction,
    ) {
        for i in 0..learn_data.node_values.len() {
            let dcost = cost_fn.call_d(learn_data.activation_values[i], expected_outputs[i]);
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
        cost_fn: CostFunction,
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

        let remainder_activation = chunks_activation.remainder();
        let remainder_weighted = chunks_weighted.remainder();
        let remainder_expected = chunks_expected.remainder();

        for ((chunk_activation, chunk_weighted), (chunk_expected, chunk_node_vals)) in
            chunks_activation
                .zip(chunks_weighted)
                .zip(chunks_expected.zip(chunks_node_vals.by_ref()))
        {
            use crate::zneural_network::cost::mse_d_simd;

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
